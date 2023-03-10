from engine.commons.common.simulator_configurator import SimulatorConfigurator
from engine.commons.common_data.base_info import PathInfo
from domain.site_progress import SiteProgress
from commons.parameters import Parameters
from domain.site_models import SiteModels
from engine.commons.data_inventory.comDataInv import ComDataInventory
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import os
import numpy as np


def start(dataset_id: str, simulation_prefix: str):
    Parameters.set_engine_database()

    SimulatorConfigurator.configurate_simulator(dataset_id=dataset_id, simulation_prefix=simulation_prefix)
    SimulatorConfigurator.run_simulator()
    # ORP 계산 결과
    # WH 보시면 Prod 변환 기록이 나올겁니다.
    # whID: warehouse id, currProd: PS.FROM_PROD_CODE, nextProd: PS.TO_PROD_CODE
    t1 = time.time()
    try1 = Analysis()
    try1.analysis_engine_result("target")
    t2 = time.time()
    print(t2-t1)

class Analysis:

    def __init__(self, *args):
        # args에 입력된 lot_id에 해당하는 lot들의 history만 출력합니다.
        # args를 입력하지 않으면 모든 lot들이 출력됩니다.
        if args:
            self.id_list = [lot_id for lot_id in args]
        else:
            lots = SiteModels.lots
            self.id_list = [lot_id for lot_id in lots.keys()]

        # 메인 데이터 프레임입니다.
        self.df = None

        # target을 기준으로 color를 나눈 figure입니다
        self.fig_target_col = None
        # event를 기준으로 color를 나눈 figure
        self.fig_event_col = None

        # lot_id별 가장 늦게 끝나는 시간입니다
        self.latest_finish = {}
        # 공장 가동 비율입니다.
        self.facility_utiliztion_rate = {}
        # orp 계산 결과의 동선
        self.node = {}


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
    def modify_hov_name_from_cus_data(bar):
        """
        hover name을 customdata의 data로 설정합니다.
        """
        bar.hovertext = bar.customdata

    @staticmethod
    def modify_legendgroup(bar, name):
        """
        범례 이름을 원하는 name으로 변환
        """
        bar.legendgroup = name
        bar.name = name


    @staticmethod
    def str_to_datetime(time_data: int) -> datetime:
        """
        str형태의 start_time 파라미터를 초의 형태인 time_data 파라미터와 연산 후
        datetime 형태의 데이터를 return 합니다.
        """
        # 날짜 연산을 위해 문자열을 datetime형태로
        temp_date = datetime.strptime(SiteProgress.plan_start_time , "%Y-%m-%d %H:%M:%S")

        # datetime형태를 초 형태로 변환
        start_date = time.mktime(temp_date.timetuple())

        # 초 데이터끼리 연산 후 datetime으로 변환
        time_datetime = datetime.fromtimestamp(start_date + time_data)

        return time_datetime

    def generate_dataframe(self):
        """
        df(데이터프레임)를 생성합니다.
        column(lot_id, target, event, start, finish, lpst)
        """

        lots = SiteModels.lots
        lot_history_data = []
        late_lot_history = []

        for lot_ids in self.id_list:
            histories = lots[lot_ids].history

            # lot별 가장 빠른 시작 시간을 임시 저장합니다.
            start_d = 0
            # lot별 가장 늦은 종료 시간을 임시 저장합니다.
            finish_d = 0
            # lot별 track_in_event가 발생한 시간을 임시 저장합니다.
            sum_track_in_time = 0

            for history in histories:
                if start_d == 0:
                    start_d = history[3]
                elif history[2] < start_d:
                    start_d = history[2]

                elif history[3] > finish_d:
                    finish_d = history[3]

                # start 데이터를 datetime형태의 데이터로 바꿉니다.
                start_datetime = Analysis.str_to_datetime( history[2])
                finish_datetime = Analysis.str_to_datetime( history[3])
                lpst_datetime = Analysis.str_to_datetime( history[5])


                # 그래프로 표현할 lot의 history 결과를 출력합니다.
                lot_history_data.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                             start=start_datetime, finish=finish_datetime, lpst=lpst_datetime))

                if history[1] == "TRACK_IN":
                    sum_track_in_time += history[3] - history[2]

                # TRACK_IN event의 지연됨 표현은 하지 않기 때문에 elif를 사용합니다.
                # lpst보다 start가 더 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif lpst_datetime < start_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                     start=start_datetime, finish=finish_datetime, lpst=lpst_datetime))

                # lpst보다 start가 더 빠르지만 finish보다 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif start_datetime <= lpst_datetime <= finish_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                     start=lpst_datetime, finish=finish_datetime, lpst=lpst_datetime))

            # lot_id를 key, lot_id별 가장 늦은 시간을 value로 하는 딕셔너리 생성
            self.latest_finish[lot_ids] = finish_d

            # lot_id를 key, lot_id별 가장
            self.facility_utiliztion_rate[lot_ids] = sum_track_in_time * 100 / (finish_d - start_d)

        # 지연된 정보를
        lot_history_data.extend(late_lot_history)
        self.df = pd.DataFrame(lot_history_data)


    def update_layout_target(self):
        """
        figure(fig_target_col)의 layout 항목을 수정합니다.
        """
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
        target을 기준으로 color를 나눈 figure를 작성합니다.
        """
        self.fig_target_col = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target",
                                          color="target", hover_name="event", hover_data=["lpst"])
        self.update_layout_target()

    def update_layout_event(self):
        """
        figure(fig_target_col)의 layout 항목을 수정합니다.
        """
        anno_fact_util = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 8000), y=lot_id,
                               text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%", showarrow=False) for lot_id in self.id_list]

        with self.fig_target_col.batch_update():
            self.fig_event_col.layout["annotations"] = anno_fact_util

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
            [Analysis.modify_hov_name_from_cus_data(bar) for bar in self.fig_event_col.data if
             ('late' in bar.legendgroup)]

    def generate_fig_event_col(self):
        """
        event를 기준으로 color를 나눈 figure를 생성합니다.
        """
        event_list =self.df['event'].unique()
        col_list = ['#20B2AA', 'grey', 'limegreen', '#F8C471' ,'forestgreen']
        col_dict = {event_list[i]:col_list[i] for i in range(len(event_list))}

        self.fig_event_col = px.timeline(self.df, x_start='start', x_end="finish", y="lot_id", text="target",
                                         hover_name="target",
                                         color="event", color_discrete_map=col_dict, hover_data=["lpst"])
        self.update_layout_event()

    def append_track_in_df(self):
        """
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(track_in_df, ignore_index=True)

    def generate_sankey_dataframe(self):
        node_data = [dict(source=pro.currProd, target=pro.nextProd, value=1) for pro in
                      ComDataInventory.orpDataInv.orpDepthInfoList if pro.nextProd != pro.currProd]
        self.dff = pd.DataFrame(node_data)
        self.dddd = self.dff
        source_list = self.dff.source.unique()
        targ_list = self.dff.target.unique()
        self.label_name = list(set(np.append(source_list,targ_list)))
        for i in range(len(self.label_name)):
            self.dff = self.dff.replace(self.label_name[i], i)


    def generate_fig_sankey_diagram(self):
        self.fig_sankey_diagram = go.Figure(data=[go.Sankey(
            valueformat=".0f",
            valuesuffix="obj",
            # Define nodes
            node=dict(
                label=self.label_name,
            ),
            # Add links
            link=dict(
                source=self.dff.source,
                target=self.dff.target,
                value=self.dff.value,
            ))])

        self.fig_sankey_diagram.update_layout(
            title_text="sankey diagram",
            font_size=10)

    def analysis_engine_result(self, *args):
        """
        engine의 결과값을 그래프로 시각화 해줍니다.
        """
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


        if 'sankey' in args:
            self.generate_sankey_dataframe()
            self.generate_fig_sankey_diagram()
            self.fig_sankey_diagram.write_html(f"{PathInfo.xlsx}{os.sep}temp_sankey.html", default_width=1800, default_height=900)
            self.fig_sankey_diagram.show(width=2000, height=750)