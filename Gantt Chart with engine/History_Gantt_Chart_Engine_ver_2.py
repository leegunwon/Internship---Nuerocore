import os
from commons.parameters import Parameters
from engine.commons.common.simulator_configurator import SimulatorConfigurator
from datetime import datetime
from engine.commons.common_data.base_info import PathInfo
from domain.site_models import SiteModels
from domain.site_progress import SiteProgress
import pandas as pd
import plotly.express as px
import time

def start(dataset_id: str, simulation_prefix: str):
    Parameters.set_engine_database()

    SimulatorConfigurator.configurate_simulator(dataset_id=dataset_id, simulation_prefix=simulation_prefix)
    t0 = time.time()
    SimulatorConfigurator.run_simulator()
    t1 = time.time()
    lots = SiteModels.lots
    try1 = Analysis()
    try1.analysis_engine_result()
    t2 = time.time()
    print((t2 - t1)/(t1 - t0))
    print(t2 - t1)
    print(t1 - t0)

class Analysis:

    def __init__(self, *args):

        # args는 lot_id를 어떤 범위 슬라이싱 할것인지, 1은 처음 lot_id, 2는 마지막 lot_id
        if args:
            x = iter(args)
            self.lot_id1 = next(x)
            self.lot_id2 = next(x)

        #finish - start로 만든 operation_time(실제 동작 시간)를 모두 더해서 sum_operation_time을 제작
        self.total_operation_time = {}

        # 공장 가동 비율이다.
        self.facility_utiliztion_rate = {}


        # track_in 공정만 모두 더한 값 제작
        self.track_in_operation_time = {}
        # 가장 빠른 시작 시간, 가장 늦게 끝나는 시간
        self.id_start = {}
        self.id_finish = {}

    @staticmethod
    def modify_width(bar, width):
        bar.width = width

    @staticmethod
    def modify_opacity(bar, opacity):
        bar.opacity = opacity

    @staticmethod
    def modify_color(bar, color):
        bar.marker['color'] = color

    @staticmethod
    def modify_line(bar, color):
        bar.marker['line'] = color


    @staticmethod
    def generate_timedata(operation_time) -> datetime:
        """
        기존 데이터 형식이 초 형태로 되어있는 history의 시간들과 날짜 형식인 초기 날짜를 연산하기 위해
        날짜 형태를 초 형태로 변환 시켜서 연산후 다시 날짜 형식으로 변환해서 리턴하는 함수
        """
        temp_date = datetime.strptime(SiteProgress.plan_start_time, "%Y-%m-%d %H:%M:%S")
        start_date = time.mktime(temp_date.timetuple())

        titime = datetime.fromtimestamp(start_date + operation_time)


        return titime

    def generate_dataframe(self):
        """
        딕셔너리 형태인 lots를 불러와서 lot_id와 그 히스토리를 리스트로 정리하고,
        그 리스트 안에 요소들을 딕셔너리로 id와 히스토리들을 lot_id, target, event, start, finish, lpst와 키, 벨류 형태로 저장
        """
        lots = SiteModels.lots
        lot_gantt_history = []
        lateness_lot_gantt = []
        self.id_list = [lot_id for lot_id in lots.keys()]
        # instance
        try:
            # 1. 값 입력 받아서 일부분 출력하게 데이터 슬라이싱 lot_id 받아서 검색  temp 리스트에서 위치를 알수 있는지 확인
            pos_id1 = self.id_list.index(self.lot_id1)
            pos_id2 = self.id_list.index(self.lot_id2)
            self.id_list = self.id_list[pos_id1:pos_id2]
        except:
            pass

        for lot_id in self.id_list:
            histories = lots[lot_id].history
            start_d = 0
            finish_d = 0
            sum_operation_time = 0

            for history in histories:

                if history[2] < start_d:
                    start_d = history[2]
                else:
                    pass

                if finish_d == 0:
                    finish_d = history[3]
                if history[3] > finish_d:
                    finish_d = history[3]
                else:
                    pass

                start_t = Analysis.generate_timedata(history[2])
                finish_t = Analysis.generate_timedata(history[3])
                lpst_t = Analysis.generate_timedata(history[5])
                 # history[3] - history[2]로 operation time list 제작
                operation_t = history[3] - history[2]

                if history[1] == "TRACK_IN":
                     sum_operation_time += operation_t

                # lpst가 start time보다 빠르면 문제 있는거
                # 5는 lpst 2는 start
                # lpst가 start와 finish 사이

                if operation_t == 0:
                     lot_gantt_history.append(dict(lot_id=lot_id, target=history[0], event=history[1],
                                                   start=start_t, finish= finish_t))
                else:
                     if history[5] < history[2]:
                          lot_gantt_history.append(dict(lot_id=lot_id, target=history[0], event=history[1],
                                                       start=start_t, finish=finish_t))
                          if history[1] != 'TRACK_IN':
                            lateness_lot_gantt.append(dict(lot_id=lot_id, target=" ", event=lpst_t,
                                                       start=start_t, finish=finish_t))
                     elif history[5] > history[3]:
                          lot_gantt_history.append(dict(lot_id=lot_id, target=history[0], event=history[1],
                                                        start=start_t, finish=finish_t,))

                     elif history[2] <= history[5] <= history[3]:
                         lot_gantt_history.append(dict(lot_id=lot_id, target=history[0], event=history[1],
                                                       start=start_t, finish=finish_t))
                         if history[1] != 'TRACK_IN':
                            lateness_lot_gantt.append(dict(lot_id=lot_id, target=' ', event=lpst_t,
                                                        start=lpst_t, finish=finish_t))
            self.id_start[lot_id] = start_d
            self.id_finish[lot_id] = finish_d
            self.track_in_operation_time[lot_id] = sum_operation_time
            for lot_id, start in self.id_start.items():
                self.total_operation_time[lot_id] = self.id_finish[lot_id] - start

        lot_gantt_history.extend(lateness_lot_gantt)
        self.df = pd.DataFrame(lot_gantt_history)


    def generate_fig(self):
        """
        생성된 데이터 프레임을 기반으로 figure 작성
        x축은 시간, y축은 lot_id, text, color는 타겟, 큰글씨의 텍스트는 이벤트로 작성하였다.
         figure의 데이터를 기반으로 event가 TRACK_IN인 것들의 너비와 투명도를 조절하였다.
        """

        self.fig = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target", color="target",
                                hover_name="event")

        for lot_id in self.id_list:
            self.fig.add_annotation(x=Analysis.generate_timedata(self.id_finish[lot_id]+ 20000), y=lot_id,
                text=str(self.facility_utiliztion_rate[lot_id])[0:2] + str('%'),
                showarrow=False)


        self.fig.update_layout(bargap=0.1, width=1500, uniformtext_minsize=8, uniformtext_mode='hide')
        self.fig.update_traces(textfont_size=12, textangle=270, textposition='inside', offset=False)



        [Analysis.modify_width(bar, 0.5) for bar in self.fig.data if
         ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]
        [Analysis.modify_opacity(bar, 0.4) for bar in self.fig.data if
         ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]
        [Analysis.modify_color(bar, 'grey') for bar in self.fig.data if
         (' ' in bar.legendgroup)]
        [Analysis.modify_opacity(bar, 0.6) for bar in self.fig.data if
         (' ' in bar.legendgroup)]

    def analysis_engine_result(self):
        self.generate_dataframe()
        for lot_id, track_in in self.track_in_operation_time.items():
            self.facility_utiliztion_rate[lot_id] = track_in /self.total_operation_time[lot_id] * 100
        self.track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(self.track_in_df, ignore_index=True)
        self.generate_fig()
        self.fig.write_html(f"{PathInfo.xlsx}{os.sep}temp.html")