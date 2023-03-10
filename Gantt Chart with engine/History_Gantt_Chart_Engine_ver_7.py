from engine.commons.common.simulator_configurator import SimulatorConfigurator
from engine.commons.common_data.base_info import PathInfo
from domain.site_progress import SiteProgress
from commons.parameters import Parameters
from domain.site_models import SiteModels
from datetime import datetime
import plotly.express as px
import pandas as pd
import time
import os


def start(dataset_id: str, simulation_prefix: str):
    Parameters.set_engine_database()

    SimulatorConfigurator.configurate_simulator(dataset_id=dataset_id, simulation_prefix=simulation_prefix)
    SimulatorConfigurator.run_simulator()


    try1 = Analysis()
    try1.analysis_engine_result('target')

class Analysis:

    # temp_strptime = datetime.strptime()
    # temp_mktime = time.mktime()
    # temp_fromtimestamp = datetime.fromtimestamp()
    def __init__(self, *args):
        # args는 보고 싶은 lot_id의 범위를 정할 수 있습니다.
        if args:
            x = iter(args)
            self.lot_id1 = next(x)
            self.lot_id2 = next(x)

        self.fig_event = None
        # 공장 가동 비율입니다.
        self.facility_utiliztion_rate = {}

        # 메인 데이터 프레임입니다.
        self.df = None
        # track_in 이벤트만 따로 모아놓은 데이터 프레임입니다.

        # 메인 피규어입니다
        self.fig_target = None
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
    def modify_hover_name(bar):
        bar.hovertext = bar.customdata

    @staticmethod
    def modify_legendgroup(bar, name):
        bar.legendgroup = name
        bar.name = name


    @staticmethod
    def str_to_datetime(operation_time) -> datetime:
        """
        기존 데이터 형식이 초 형태로 되어있는 history의 시간들과 날짜 형식인 초기 날짜를 연산하기 위해
        날짜 형태를 초 형태로 변환 시켜서 연산후 다시 날짜 형식으로 변환해서 리턴하는 함수
        """

        temp_date = datetime.strptime(SiteProgress.plan_start_time, "%Y-%m-%d %H:%M:%S")
        start_date = time.mktime(temp_date.timetuple())

        time_datetime = datetime.fromtimestamp(start_date + operation_time)

        return time_datetime

    def generate_dataframe(self):
        """
        lot_id들을 리스트에 정리하고,
        그 리스트 안에 요소들을 딕셔너리로 id와 히스토리들을 lot_id, target, event, start, finish 키, 벨류 형태로 저장
        """
        lots = SiteModels.lots
        lot_gantt_history = []
        lateness_lot_gantt = []
        # 전체 가동 시간입니다.
        # track_in event 시간 합입니다.
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
                start_datetime = Analysis.str_to_datetime(history[2])
                finish_datetime = Analysis.str_to_datetime(history[3])
                lpst_datetime = Analysis.str_to_datetime(history[5])
                operation_t = history[3] - history[2]

                if history[1] == "TRACK_IN":
                    sum_operation_time += operation_t

                if history[5] < history[2]:
                    lot_gantt_history.append(dict(Task=lot_ids, target=history[0], event=history[1],
                                                  Start=start_datetime, Finish=finish_datetime, lpst=lpst_datetime))
                    if history[1] != 'TRACK_IN':
                        lateness_lot_gantt.append(dict(Task=lot_ids, target=' ', event='late',
                                                       Start=start_datetime, Finish=finish_datetime, lpst=lpst_datetime))
                elif history[5] > history[3]:
                    lot_gantt_history.append(dict(Task=lot_ids, target=history[0], event=history[1],
                                                  Start=start_datetime, Finish=finish_datetime, lpst=lpst_datetime))

                elif history[2] <= history[5] <= history[3]:
                    lot_gantt_history.append(dict(Task=lot_ids, target=history[0], event=history[1],
                                                  Start=start_datetime, Finish=finish_datetime, lpst=lpst_datetime))
                    if history[1] != 'TRACK_IN':
                        lateness_lot_gantt.append(dict(Task=lot_ids, target=' ', event='late',
                                                       Start=lpst_datetime, Finish=finish_datetime, lpst=lpst_datetime))
            # id_finish는 가장 늦은 finish_time
            self.id_finish[lot_ids] = finish_d
            #설비 가동률
            self.facility_utiliztion_rate[lot_ids] = sum_operation_time*100 / (finish_d - start_d)


        lot_gantt_history.extend(lateness_lot_gantt)
        self.df = pd.DataFrame(lot_gantt_history)

    def generate_fig_target(self):
        """
        생성된 데이터 프레임(self.df)을 기반으로 figure 작성
        """
        # template list ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]

        self.fig_target = px.timeline(self.df, x_start='Start', x_end="Finish", y="Task", text="target",
                                      color="target", hover_name="event", labels={'Task':'lot_id'}                                      )

        # 파일 저장에는 width 2300이 적당하지만 show에서는 2000이 적당하다
        # height=900, height=750
        add_layout = [dict(x=Analysis.str_to_datetime(self.id_finish[lot_id] + 13000), y=lot_id,
                           text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%", showarrow=False)
                      for lot_id in self.id_list]

        self.fig_target.layout["annotations"] = add_layout


        # TRACK_IN event의 너비와 투명도를 조정하였습니다.
        [Analysis.modify_width(bar, 0.5) for bar in self.fig_target.data if
         ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]
        [Analysis.modify_opacity(bar, 0.4) for bar in self.fig_target.data if
         ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]

        # lpst를 충족하지 못한 부분을 회색으로 바꾸고 투명도를 조정했습니다.
        [Analysis.modify_color(bar, 'black') for bar in self.fig_target.data if
         (' ' in bar.legendgroup)]
        [Analysis.modify_opacity(bar, 0.5) for bar in self.fig_target.data if
         (' ' in bar.legendgroup)]

    def generate_fig_event(self):
        # event별 색상을 지정하였습니다.
        event_list =list(set([events for events in self.df['event'].values if type(events) == str]))
        event_list.sort()

        col_list = ['#20B2AA', 'yellow', 'limegreen', '#F8C471' ,'forestgreen']
        col_dict = {event_list[i]:col_list[i] for i in range(len(event_list))}

        self.fig_event = px.timeline(self.df, x_start='Start', x_end="Finish", y="Task", text="target",
                                     hover_name="target", labels={'Task':'lot_id'},
                                     color="event", color_discrete_map=col_dict, hover_data=["lpst"])

        add_layout = [dict(x=Analysis.str_to_datetime(self.id_finish[lot_id] + 13000), y=lot_id,
                           text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%",showarrow=False)
                      for lot_id in self.id_list]

        self.fig_event.layout["annotations"] = add_layout


        self.fig_event.update_layout(plot_bgcolor='#D6DBDF')
        # self.fig_event.update_traces(textfont_size=12, textangle=270, textposition='inside')

        [Analysis.modify_color(bar, '#7FB3D5') for bar in self.fig_event.data if
        ('WH' in bar.legendgroup)]
        [Analysis.modify_legendgroup(bar, 'WH') for bar in self.fig_event.data if
        ('WH' in bar.legendgroup)]

        [Analysis.modify_width(bar, 0.5) for bar in self.fig_event.data if
         ('TRACK_IN' in bar.legendgroup)]
        [Analysis.modify_opacity(bar, 0.7) for bar in self.fig_event.data if
         ('TRACK_IN' in bar.legendgroup)]

        [Analysis.modify_color(bar, 'black') for bar in self.fig_event.data if
        ('late' in bar.legendgroup)]
        [Analysis.modify_opacity(bar, 0.4) for bar in self.fig_event.data if
        ('late' in bar.legendgroup)]
        [Analysis.modify_hover_name(bar) for bar in self.fig_event.data if
        ('late' in bar.legendgroup)]

    def analysis_engine_result(self, *args):
        self.generate_dataframe()

        # track_in 이벤트만 따로 모아놓은 데이터 프레임입니다.
        track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(track_in_df, ignore_index=True)

        if 'target' in args:
            self.generate_fig_target()
            self.fig_target.write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
            self.fig_target.show(width=2000, height=750)

        if 'event' in args:
            self.generate_fig_event()
            self.fig_event.write_html(f"{PathInfo.xlsx}{os.sep}temp_event.html", default_width=2300, default_height=900)
            self.fig_event.show(width=2000, height=750)


