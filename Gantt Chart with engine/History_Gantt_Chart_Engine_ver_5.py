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


    try1 = Analysis("V_F_00001", "V_F_00010")
    try1.analysis_engine_result('target', "event")

class Analysis:

    def __init__(self, *args):
        # args는 특정 lot_id만 보고 싶을 때 lot_id를 지정합니다.
        if args:
            self.id_list = [lot_id for lot_id in args]
        else:
            lots = SiteModels.lots
            self.id_list = [lot_id for lot_id in lots.keys()]

        # 공장 가동 비율입니다.
        self.facility_utiliztion_rate = {}

        # 메인 데이터 프레임입니다.
        self.df = None
        # 타겟, 이벤트 피규어입니다
        self.fig_target_col = None
        self.fig_event_col = None
        # 가장 빠른 시작 시간, 가장 늦게 끝나는 시간입니다
        self.latest_finish = {}


    @staticmethod
    def modify_width(bar, width):
        """
        막대의 너비를 설정합니다.
        """
        bar.width = width

    @staticmethod
    def modify_opacity(bar, opacity):
        """
        그래프 막대의 투명도를 설정합니다
        """
        bar.opacity = opacity

    @staticmethod
    def modify_color(bar, color):
        """
        bar marker의 색을 지정한 색으로 설정합니다
        """
        bar.marker['color'] = color

    @staticmethod
    def modify_hov_name_from_cus_data(bar, indexing):
        """
        hover name을 customdata의 data로 설정합니다.
        """
        bar.hovertext = bar.customdata[indexing]

    @staticmethod
    def modify_legendgroup(bar, name):
        """
        범례 이름을 원하는 name으로 변환
        """
        bar.legendgroup = name
        bar.name = name


    @staticmethod
    def str_to_datetime(time_data) -> datetime:
        """
        기존 데이터 형식이 초 형태로 되어있는 history의 시간들과 날짜 형식인 초기 날짜를 연산하기 위해
        날짜 형태를 초 형태로 변환 시켜서 연산후 다시 날짜 형식으로 변환해서 리턴하는 함수
        """
        # 날짜 연산을 위해 문자열을 datetime형태로
        temp_date = datetime.strptime(SiteProgress.plan_start_time, "%Y-%m-%d %H:%M:%S")

        # datetime형태를 초 형태로 변환
        start_date = time.mktime(temp_date.timetuple())

        # 초 데이터끼리 연산 후 datetime으로 변환
        time_datetime = datetime.fromtimestamp(start_date + time_data)

        return time_datetime

    def generate_dataframe(self):
        """
        lot_id들을 리스트에 정리하고,
        그 리스트 안에 요소들을 딕셔너리로 id와 히스토리들을 lot_id, target, event, start, finish 키, 벨류 형태로 저장
        """

        lots = SiteModels.lots
        lot_history_data = []
        late_lot_history = []


        for lot_ids in self.id_list:
            histories = lots[lot_ids].history
            start_d = 0
            finish_d = 0
            sum_operation_time = 0

            for history in histories:
                if start_d == 0:
                    start_d = history[3]
                elif history[2] < start_d:
                    start_d = history[2]

                elif history[3] > finish_d:
                    finish_d = history[3]


                start_datetime = Analysis.str_to_datetime(history[2])
                finish_datetime = Analysis.str_to_datetime(history[3])
                lpst_datetime = Analysis.str_to_datetime(history[5])


                if history[1] == "TRACK_IN":
                    sum_operation_time += history[3] - history[2]

                elif lpst_datetime < start_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                     start=start_datetime, finish=finish_datetime, lpst=lpst_datetime))

                elif start_datetime <= lpst_datetime <= finish_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                     start=lpst_datetime, finish=finish_datetime, lpst=lpst_datetime))

                lot_history_data.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                             start=start_datetime, finish=finish_datetime, lpst=lpst_datetime))

            self.latest_finish[lot_ids] = finish_d
            self.facility_utiliztion_rate[lot_ids] = sum_operation_time * 100 / (finish_d - start_d)

        lot_history_data.extend(late_lot_history)
        self.df = pd.DataFrame(lot_history_data)


    def update_layout_target(self):
        anno_factory_utilization = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 6000), y=lot_id,
                               text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%", showarrow=False)
                          for lot_id in self.id_list]

        # self.fig_target_col.layout.hovermode = "x"    # hovermode입니다 x축 기준 동일 선상에 있는 모든 hover가 출력됩니다.
        with self.fig_target_col.batch_update():
            self.fig_target_col.layout["annotations"] = anno_factory_utilization
            # TRACK_IN event의 너비와 투명도를 조정하였습니다.
            [Analysis.modify_width(bar, 0.5) for bar in self.fig_target_col.data if
             ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]
            [Analysis.modify_opacity(bar, 0.4) for bar in self.fig_target_col.data if
             ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]
            # lpst를 충족하지 못한 부분을 회색으로 바꾸고 투명도를 조정했습니다.
            [Analysis.modify_color(bar, 'black') for bar in self.fig_target_col.data if
             (' ' in bar.legendgroup)]
            [Analysis.modify_opacity(bar, 0.6) for bar in self.fig_target_col.data if
             (' ' in bar.legendgroup)]
            [Analysis.modify_hov_name_from_cus_data(bar) for bar in self.fig_target_col.data if
             (' ' in bar.legendgroup)]

    def generate_fig_target_col(self):
        """
        생성된 데이터 프레임(self.df)을 기반으로 figure 작성
        """
        self.fig_target_col = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target",
                                          color="target", hover_name="event", hover_data=["lpst"])
        self.update_layout_target()

    def update_layout_event(self):
        anno_fact_util = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 8000), y=lot_id,
                               text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%", showarrow=False) for lot_id in self.id_list]

        with self.fig_target_col.batch_update():
            self.fig_event_col.layout["annotations"] = anno_fact_util
            self.fig_event_col.update_layout(plot_bgcolor='#D6DBDF')

            [Analysis.modify_color(bar, '#7FB3D5') for bar in self.fig_event_col.data if
             ('WH' in bar.legendgroup)]
            [Analysis.modify_legendgroup(bar, 'WH') for bar in self.fig_event_col.data if
             ('WH' in bar.legendgroup)]

            [Analysis.modify_width(bar, 0.5) for bar in self.fig_event_col.data if
             ('TRACK_IN' in bar.legendgroup)]
            [Analysis.modify_opacity(bar, 0.7) for bar in self.fig_event_col.data if
             ('TRACK_IN' in bar.legendgroup)]

            [Analysis.modify_color(bar, 'black') for bar in self.fig_event_col.data if
             ('late' in bar.legendgroup)]
            [Analysis.modify_opacity(bar, 0.5) for bar in self.fig_event_col.data if
             ('late' in bar.legendgroup)]
            [Analysis.modify_hov_name_from_cus_data(bar, 0) for bar in self.fig_event_col.data if
             ('late' in bar.legendgroup)]

    def generate_fig_event_col(self):
        event_list =self.df['event'].unique()
        col_list = ['#20B2AA', 'grey', 'limegreen', '#F8C471' ,'forestgreen']
        col_dict = {event_list[i]:col_list[i] for i in range(len(event_list))}

        self.fig_event_col = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target",
                                     hover_name="target",
                                     color="event", color_discrete_map=col_dict, hover_data=["lpst"])
        self.update_layout_event()

    def append_track_in_df(self):
        """
        track_in event를
        """
        track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(track_in_df, ignore_index=True)

    def analysis_engine_result(self, *args):
        self.generate_dataframe()
        # track_in 이벤트만 따로 모아놓은 데이터 프레임입니다.
        self.append_track_in_df()

        if 'target' in args:
            self.generate_fig_target_col()
            self.fig_target_col.write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
            self.fig_target_col.show(width=2000, height=750, config={'modeBarButtonsToAdd':['drawline', 'drawopenpath', 'drawclosedpath',
                                                                                            'drawcircle', 'drawrect', 'eraseshape']})

        if 'event' in args:
            self.generate_fig_event_col()
            self.fig_event_col.write_html(f"{PathInfo.xlsx}{os.sep}temp_event.html", default_width=2300, default_height=900)
            self.fig_event_col.show(width=2000, height=750)

