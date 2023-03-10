from engine.commons.common.simulator_configurator import SimulatorConfigurator
from engine.commons.common_data.base_info import PathInfo
from domain.site_progress import SiteProgress
from commons.parameters import Parameters
from domain.site_models import SiteModels
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import os


def start(dataset_id: str, simulation_prefix: str):
    Parameters.set_engine_database()

    SimulatorConfigurator.configurate_simulator(dataset_id=dataset_id, simulation_prefix=simulation_prefix)
    SimulatorConfigurator.run_simulator()


    try1 = Analysis()
    try1.analysis_engine_result()


class Analysis:

    def __init__(self, *args):
        # args는 보고 싶은 lot_id의 범위를 정할 수 있습니다.
        if args:
            x = iter(args)
            self.lot_id1 = next(x)
            self.lot_id2 = next(x)

        self.fig_sub = None
        # 공장 가동 비율입니다.
        self.facility_utiliztion_rate = {}

        # 메인 데이터 프레임입니다.
        self.df = None
        # track_in 이벤트만 따로 모아놓은 데이터 프레임입니다.

        # 메인 피규어입니다
        self.fig = None
        # id별 track_in 공정만 모두 더한 값입니다.

        # 가장 빠른 시작 시간, 가장 늦게 끝나는 시간입니다
        self.id_finish = {}
        # lot의 id 들을 리스트로 저장했습니다.
        self.id_list =[]
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
    def modify_line_color(bar, color):
        go.bar.marker['line'] = color

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
        lot_id들을 리스트에 정리하고,
        그 리스트 안에 요소들을 딕셔너리로 id와 히스토리들을 lot_id, target, event, start, finish 키, 벨류 형태로 저장
        """
        lots = SiteModels.lots
        lot_gantt_history = []
        lateness_lot_gantt = []
        id_start = {}
        # 전체 가동 시간입니다.
        # track_in event 시간 합입니다.
        total_operation_time = {}
        track_in_operation_time = {}
        self.id_list = [lot_id for lot_id in lots.keys()]

        # 1. 값 입력 받아서 일부분 출력하게 데이터 슬라이싱 lot_id 받아서 검색  temp 리스트에서 위치를 알수 있는지 확인
        try:
            pos_id1 = self.id_list.index(self.lot_id1)
            pos_id2 = self.id_list.index(self.lot_id2)
            self.id_list = self.id_list[pos_id1:pos_id2]
        except AttributeError:
            pass

        for lot_ids in self.id_list:
            histories = lots[lot_ids].history
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

                # history의 요소들을 datetime으로 변환 시켜 변수에 저장했습니다.
                start_t = Analysis.generate_timedata(history[2])
                finish_t = Analysis.generate_timedata(history[3])
                lpst_t = Analysis.generate_timedata(history[5])
                operation_t = history[3] - history[2]

                if history[1] == "TRACK_IN":
                    sum_operation_time += operation_t

                # zero division error를 회피하게 만든 조건입니다.
                if operation_t == 0:
                    lot_gantt_history.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                                  start=start_t, finish=finish_t))
                else:
                    if history[5] < history[2]:
                        lot_gantt_history.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                                      start=start_t, finish=finish_t))
                        if history[1] != 'TRACK_IN':
                            lateness_lot_gantt.append(dict(lot_id=lot_ids, target=" ", event=lpst_t,
                                                           start=start_t, finish=finish_t))
                    elif history[5] > history[3]:
                        lot_gantt_history.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                                      start=start_t, finish=finish_t))

                    elif history[2] <= history[5] <= history[3]:
                        lot_gantt_history.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                                      start=start_t, finish=finish_t))
                        if history[1] != 'TRACK_IN':
                            lateness_lot_gantt.append(dict(lot_id=lot_ids, target=' ', event=lpst_t,
                                                           start=lpst_t, finish=finish_t))

            # id_start list는 가장 이른 start_time
            # id_finish는 가장 늦은 finish_time
            id_start[lot_ids] = start_d
            self.id_finish[lot_ids] = finish_d
            track_in_operation_time[lot_ids] = sum_operation_time
            for lot_id1, start1 in id_start.items():
                total_operation_time[lot_id1] = self.id_finish[lot_id1] - start1
            for lot_id, track_in in track_in_operation_time.items():
                self.facility_utiliztion_rate[lot_id] = track_in / total_operation_time[lot_id] * 100


        lot_gantt_history.extend(lateness_lot_gantt)
        self.df = pd.DataFrame(lot_gantt_history)

    def generate_fig(self):
        """
        생성된 데이터 프레임(self.df)을 기반으로 figure 작성
        """
        t1 = time.time()
        # self.fig = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target", color="target",
        #                        hover_name="event")
        event_list =list(set([events for events in self.df['event'].values if type(events) == str]))
        event_list.sort()
        col_list = ['#52BE80', '#F8C471', '#2E8B57', '#AF7AC5', '#E6B0AA']
        col_dict = {event_list[i]:col_list[i] for i in range(len(event_list))}
        self.fig_sub = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target", color="event",
                                   hover_name="target", color_discrete_map = col_dict)
        # for lot_id in self.id_list:
        #     self.fig.add_annotation(x=Analysis.generate_timedata(self.id_finish[lot_id] + 20000), y=lot_id,
        #                             text=str(self.facility_utiliztion_rate[lot_id])[0:2] + str('%'),
        #                             showarrow=False)
        #
        # self.fig.update_layout(bargap=0.1, width=1500)
        # self.fig.update_traces(textfont_size=12, textangle=270, textposition='inside', offset=False)
        #
        # # TRACK_IN event의 너비와 투명도를 조정하였습니다.
        # [Analysis.modify_width(bar, 0.5) for bar in self.fig.data if
        #  ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]
        # [Analysis.modify_opacity(bar, 0.4) for bar in self.fig.data if
        #  ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]

        # # lpst를 충족하지 못한 부분을 회색으로 바꾸고 투명도를 조정했습니다.
        # [Analysis.modify_color(bar, '#17202A') for bar in self.fig.data if
        #  (' ' in bar.legendgroup)]
        # [Analysis.modify_opacity(bar, 0.6) for bar in self.fig.data if
        #  (' ' in bar.legendgroup)]

        for lot_id in self.id_list:
            self.fig_sub.add_annotation(x=Analysis.generate_timedata(self.id_finish[lot_id] + 20000), y=lot_id,
                                    text=str(self.facility_utiliztion_rate[lot_id])[0:2] + str('%'),
                                    showarrow=False)

        self.fig_sub.update_layout(bargap=0.1, width=1500, plot_bgcolor = '#D6DBDF')
        self.fig_sub.update_traces(textfont_size=12, textangle=270, textposition='inside')

        [Analysis.modify_color(bar, '#7FB3D5') for bar in self.fig_sub.data if
         ('WH' in bar.legendgroup)]
        [Analysis.modify_color(bar, '#17202A') for bar in self.fig_sub.data if
         (' ' in bar.text)]
        [Analysis.modify_opacity(bar, 0.4) for bar in self.fig_sub.data if
         (' ' in bar.text)]
        [Analysis.modify_line_color(bar,'black') for bar in self.fig_sub.data]
        t2 = time.time()
        print(t2 -t1)

    def analysis_engine_result(self):
        self.generate_dataframe()
        # track_in 이벤트만 따로 모아놓은 데이터 프레임입니다.

        track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(track_in_df, ignore_index=True)

        self.generate_fig()
        self.fig_sub.show()
      #   self.fig.write_html(f"{PathInfo.xlsx}{os.sep}temp.html")
