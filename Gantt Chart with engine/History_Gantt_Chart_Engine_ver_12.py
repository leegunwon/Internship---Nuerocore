from engine.commons.common.simulator_configurator import SimulatorConfigurator
from engine.commons.common_data.base_info import PathInfo
from domain.site_progress import SiteProgress
from commons.parameters import Parameters
from domain.site_models import SiteModels
from engine.commons.data_inventory.comDataInv import ComDataInventory
from engine.commons.util.pandas_util import PandasUtil
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import time
import os
import numpy as np


def start(dataset_id: str, simulation_prefix: str):
    Parameters.set_plan_horizon(1)
    Parameters.set_engine_database()
    t0 = time.time()
    Parameters.operation_dsp_rules = {"2100": "ORP_RES_REQ"}
    SimulatorConfigurator.configurate_simulator(dataset_id=dataset_id, simulation_prefix=simulation_prefix)
    SimulatorConfigurator.run_simulator()
    # ORP 계산 결과
    # WH 보시면 Prod 변환 기록이 나올겁니다.
    # whID: warehouse id, currProd: PS.FROM_PROD_CODE, nextProd: PS.TO_PROD_CODE
    ps_df = ComDataInventory.input_data.data_frames['in_MainPsData']
    df = PandasUtil.select(ps_df, ['FROM_PROD_CODE', 'TO_PROD_CODE'])
    t1 = time.time()
    # logging.basicConfig(level=logging.INFO)
    try1 = Analysis()
    try1.analysis_engine_result_lot_history('sankey')
    t2 = time.time()
    print(t1-t0, t2-t1)



class Analysis:

    def __init__(self, **kwargs: list):
        # args에 입력된 lot_id에 해당하는 lot들의 history만 출력합니다.
        # args를 입력하지 않으면 모든 lot들이 출력됩니다.
        self.fig_treemap = None
        self.df_treemap = None
        self.fig_sankey = None
        self.df_sankey = None
        self.df_late = None
        self.label_name = None

        if 'prod_id' in kwargs.keys():
            if kwargs['prod_id'] is not None:
                self.prodID = [prodID for prodID in kwargs['prod_id']]
            else:
                self.prodID = []
        else:
            self.prodID = []
        # logging.INFO(f"prod_id={self.prodID}")
        if 'lot_id' in kwargs.keys():
            if kwargs['lot_id'] is not None and kwargs['lot_id'] != []:
                self.id_list = [lot_id for lot_id in kwargs['lot_id']]
            else:
                self.id_list = [lot_id for lot_id in SiteModels.lots.keys()]
        else:
            self.id_list = [lot_id for lot_id in SiteModels.lots.keys()]
        # logging.INFO(f"lot_id={self.id_list}")
        # 메인 데이터 프레임입니다.
        self.df = None

        # target을 기준으로 color를 나눈 figure입니다
        self.fig_target_col = None
        # event를 기준으로 color를 나눈 figure
        self.fig_event_col = None

        # lot_id별 가장 늦게 끝나는 시간입니다
        self.latest_finish = {}
        # 공장 가동 비율입니다.
        self.facility_utilization_rate = {}
        # orp 계산 결과의 동선
        self.node = {}


    @staticmethod
    def modify_text_pos(bar, pos):
        """
        text의 위치를 bar 안쪽에 위치 시킬지 bar 바깥 쪽에 위치 시킬지 설정
        pos = ['inside', 'outside']
        """
        bar.textposition = pos


    @staticmethod
    def modify_width(bar, width):
        """
        막대의 너비를 설정합니다.
        width = (단위 px)
        """
        bar.width = width


    @staticmethod
    def modify_opacity(bar, opacity):
        """
        그래프 막대의 투명도를 설정합니다
        opacity = [0,1] 사이
        """
        bar.opacity = opacity


    @staticmethod
    def modify_color(bar, color):
        """
        bar marker의 색을 지정한 색으로 설정합니다
        color: css color code
        """
        bar.marker['color'] = color

    @staticmethod
    def modify_legendgroup(bar, name):
        """
        범례 이름을 원하는 name으로 변환
        """
        bar.legendgroup = name
        bar.name = name


    @staticmethod
    def str_to_datetime(time_data: str) -> datetime:
        """
        str형태의 start_time 파라미터를 초의 형태인 time_data 파라미터와 연산 후
        datetime 형태의 데이터를 return 합니다.
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
        df(데이터프레임)를 생성합니다.
        facility_utiliztion_rate과 latest_finish time이 생성됩니다.

        df 의 column(lot_id, target, event, start, finish, lpst)

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
                start_datetime = Analysis.str_to_datetime(history[2])
                finish_datetime = Analysis.str_to_datetime(history[3])
                lpst_datetime = Analysis.str_to_datetime(history[5])


                # 리스트 형태로 lot_history 저장
                lot_history_data.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                             start=start_datetime, operation_time=(history[3] - history[2])*1000,
                                             lpst=lpst_datetime))

                if history[1] == "TRACK_IN":
                    sum_track_in_time += history[3] - history[2]

                # TRACK_IN event의 지연됨 표현은 하지 않기 때문에 elif를 사용합니다.
                # lpst보다 start가 더 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif lpst_datetime < start_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                 start=start_datetime,  operation_time=(history[3] - history[2])*1000,
                                                 lpst=lpst_datetime))

                # lpst보다 start가 더 빠르지만 finish보다 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif start_datetime <= lpst_datetime <= finish_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                 start=lpst_datetime, operation_time=(history[3] - history[5])*1000,
                                                 lpst=lpst_datetime))

            # lot_id를 key, lot_id별 가장 늦은 시간을 value로 하는 딕셔너리 생성
            self.latest_finish[lot_ids] = finish_d

            # facility_utilization = track_in event 수행 시간 / 전체 수행 시간
            if finish_d - start_d != 0:
                self.facility_utilization_rate[lot_ids] = sum_track_in_time * 100 / (finish_d - start_d)
            else:
                self.facility_utilization_rate[lot_ids] = 0

        # 데이터 프레임 2개 생성 self.df = lot_history, self. df_late = 지연됐음을 표현하기 위한 검정 바
        self.df = pd.DataFrame(lot_history_data)
        self.df_late = pd.DataFrame(late_lot_history)


    def update_layout_target(self):
        """
        figure(fig_target_col)의 layout 항목을 수정합니다.

        anno_factory_utilization param
        x = x축 좌표 (현재 공정이 끝난 시간에 20000초를 더한 값)
        y = y축좌표 (lot_id에 해당하는 좌표)
        text = 표시할 텍스트 (lot_id별 factory_utilization 값)
        showarrow = annotation에 화살표를 표시할 것 인지 결정

        layout update
        xaxis x축 관련 정보
        tick기능:   x축 text 표시만 바꿀 수 있음, grid 기능 격자 생성가능
        rangeselector기능:   x축의 단위를 미리 버튼에 등록해서 버튼 클릭 만으로 범위 변경
        rangeslider기능:    x축의 범위를 설정할 수 있는 슬라이더 생성
        rangebreak기능:    x축 범위의 한계를 설정 할 수 있음 설정 하지 않을 경우 무한대

        barmode 그래프 바의 모드를 설정 합니다
        mode:stack: 바가 쌓아 지는 구조 (음수가 나오면 그 값에서 음의 방향으로 쌓임)
              relative: 음수는 음수 쪽으로 쌓이고 양수는 양수 쪽으로 쌓이는 stack 구조
              group: 동일한 값에 여러 개의 바가 한개의 그룹으로 묶여서 있는 구조
              overlay: 다른 모드와는 다르게 시작 지점을 원하는 곳으로 지정 가능





        """
        anno_factory_utilization = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 20000), y=lot_id,
                                         text=f"{self.facility_utilization_rate[lot_id]:.1f}%", showarrow=False)
                                    for lot_id in self.id_list]

        # self.fig_target_col.layout.hovermode = "x" x축 기준 동일 선상에 있는 모든 hover가 출력됩니다.
        self.fig_target_col.layout["annotations"] = anno_factory_utilization
        self.fig_target_col.update_layout(xaxis=dict(type='date'), barmode='overlay')

        # TRACK_IN event의 너비와 투명도를 조정하였습니다.
        [(Analysis.modify_opacity(bar, 0.8), Analysis.modify_text_pos(bar, 'outside'), Analysis.modify_width(bar, 0.5))
         for bar in self.fig_target_col.data if ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]


    def generate_fig_target_col(self):
        """
        target을 기준으로 color를 나눈 figure를 작성합니다.

        fig 생성 param
        base            시작 시간
        x               끝 시간
        y               y축 변수
        orientation     막대의 방향 (세로= default , 가로=h)
        legendgroup     레전드의 이름
        name            막대 그래프 개체 각각의 이름
        text            그래프 내부에 표시 할 데이터
        hovertemplate   호버에 데이터를 표시할 형식
        customdata       호버에 표시할 데이터
        """
        self.fig_target_col = go.Figure()

        for targ in self.df.target.unique():

            dff = self.df.loc[self.df.target == targ]
            self.fig_target_col.add_trace(
                go.Bar(base=dff.start, x=dff.operation_time, y=dff.lot_id, orientation='h',
                       xaxis='x', yaxis='y', name=targ, legendgroup=targ, text=targ,
                       hovertemplate='<b>lot_id=%{y}<br>target='+targ+'<br>start=%{base}<br>finish=%{x}<extra></extra>'))

        self.fig_target_col.add_trace(
            go.Bar(base=self.df_late.start, x=self.df_late.operation_time, y=self.df_late.lot_id, orientation='h',
                   name='late', legendgroup='late', opacity=0.6, marker=dict(color='black'),
                   customdata=self.df_late.lpst,
                   hovertemplate='<b><br>start=%{base}<br>finish=%{x}<br>lot_id=%{y}<br>lpst=%{customdata}<extra></extra>'))

        self.fig_target_col.update_traces(textposition='inside')
        self.update_layout_target()


    def update_layout_event(self):
        """
        figure(fig_target_col)의 layout 항목을 수정합니다.

        anno_factory_utilization param
        x = x축 좌표 (현재 공정이 끝난 시간에 20000초를 더한 값)
        y = y축좌표 (lot_id에 해당하는 좌표)
        text = 표시할 텍스트 (lot_id별 factory_utilization 값)

        layout update
        xaxis x축 관련 정보
         (tick기능:   x축 text 표시만 바꿀 수 있음, grid 기능 격자 생성가능
         rangeselector기능:   x축의 단위를 미리 버튼에 등록해서 버튼 클릭 만으로 범위 변경
         rangeslider기능:    x축의 범위를 설정할 수 있는 슬라이더 생성
         rangebreak기능:    x축 범위의 한계를 설정 할 수 있음 설정 하지 않을 경우 무한대
         )
        barmode 그래프 바의 모드를 설정 합니다
        (mode:stack: 바가 쌓아 지는 구조 (음수가 나오면 그 값에서 음의 방향으로 쌓임)
              relative: 음수는 음수 쪽으로 쌓이고 양수는 양수 쪽으로 쌓이는 stack 구조
              group: 동일한 값에 여러 개의 바가 한개의 그룹으로 묶여서 있는 구조
              overlay: 다른 모드와는 다르게 시작 지점을 원하는 곳으로 지정 가능
        )
        """
        anno_fact_util = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 20000), y=lot_id,
                               text=f"{self.facility_utilization_rate[lot_id]:.1f}%", showarrow=False) for lot_id in self.id_list]

        self.fig_event_col.layout["annotations"] = anno_fact_util
        self.fig_event_col.update_layout(xaxis=dict(type='date'), barmode='overlay')

        [(Analysis.modify_color(bar, '#7FB3D5'), Analysis.modify_legendgroup(bar, 'WH'))
         for bar in self.fig_event_col.data if ('WH' in bar.legendgroup)]

        [(Analysis.modify_width(bar, 0.5), Analysis.modify_opacity(bar, 0.8), Analysis.modify_text_pos(bar, 'outside'))
         for bar in self.fig_event_col.data if ('TRACK_IN' in bar.legendgroup)]


    def generate_fig_event_col(self):
        """
        event를 기준으로 color를 나눈 figure를 생성합니다.

        event가 5개 이하라는 가정하에 col_list를 적게 설정 (현재 5개 설정) event가 늘어날 경우 더 늘려 줘야 함.

        fig 생성 param
        base            시작 시간
        x               끝 시간
        y               y축 변수
        orientation     막대의 방향 (세로= default , 가로=h)
        legendgroup     레전드의 이름
        name            막대 그래프 개체 각각의 이름
        text            그래프 내부에 표시 할 데이터
        hovertemplate   호버에 데이터를 표시할 형식
        customdata       호버에 표시할 데이터

        """
        col_list = ['#20B2AA', '#F5A9D0', 'limegreen', '#F8C471', 'forestgreen']
        colors = iter(col_list)

        self.fig_event_col = go.Figure()


        for eve in self.df.event.unique():

            dff = self.df.loc[self.df.event == eve]

            self.fig_event_col.add_trace(
                go.Bar(base=dff.start, x=dff.operation_time, y=dff.lot_id, orientation='h',
                       text=dff.target, name=eve, legendgroup=eve, marker=dict(color=next(colors)),
                       hovertemplate='<b>lot_id=%{y}<br>event='+eve+'<br>start=%{base}<br>finish=%{x}<extra></extra>'))

        self.fig_event_col.add_trace(
            go.Bar(base=self.df_late.start, x=self.df_late.operation_time, y=self.df_late.lot_id, orientation='h',
                   name='late', legendgroup='late', opacsity=0.6, marker=dict(color='black'), customdata=self.df_late.lpst,
                   hovertemplate='<b><br>start=%{base}<br>finish=%{x}<br>lot_id=%{y}<br>lpst=%{customdata}<extra></extra>'))

        self.fig_event_col.update_traces(textposition='inside')
        self.update_layout_event()


    def append_track_in_df(self):
        """
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(track_in_df, ignore_index=True)


    def generate_sankey_dataframe(self, flag_treemap: bool):
        """
        현재 prod명과 다음 prod명을 source와 target에 분류하고, prod value를 1로 설정하여 데이터프레임을 작성.
        prod_ID를 숫자로 변경하여 데이터프레임에 저장합니다.
        """
        if self.prodID == []:
            data = [dict(source=prod.currProd, target=prod.nextProd, value=1)
                    for prod in ComDataInventory.orpDataInv.orpDepthInfoList if prod.nextProd != prod.currProd]

        else:
            data = [dict(source=prod.currProd, target=prod.nextProd, value=1)
                    for prod in ComDataInventory.orpDataInv.orpDepthInfoList
                    if prod.nextProd != prod.currProd if prod.dmdProdID in self.prodID]

        self.df_sankey = pd.DataFrame(data)

        if flag_treemap:
            self.df_treemap = self.df_sankey.copy()

        source_list = self.df_sankey.source.unique()
        targ_list = self.df_sankey.target.unique()

        self.label_name = list(set(np.append(source_list, targ_list)))

        for a, b in enumerate(self.label_name):
            self.df_sankey = self.df_sankey.replace(b, a)


    def generate_fig_sankey_diagram(self):
        """
        sankey_diagram
        """
        self.fig_sankey = go.Figure(data=[go.Sankey(
            # Define nodes
            node=dict(
                label=self.label_name,
                line=dict(color="black")
            ),

            # Add links
            link=dict(
                source=self.df_sankey.source,
                target=self.df_sankey.target,
                value=self.df_sankey.value,
                color='grey'
            ))])

        self.fig_sankey.update_layout(
            title_text="sankey diagram")


    def generate_treemap_dataframe(self, flag_sankey: bool):

        if not flag_sankey:
            if self.prodID == []:
                data = [dict(source=prod.currProd, target=prod.nextProd)
                        for prod in ComDataInventory.orpDataInv.orpDepthInfoList if prod.nextProd != prod.currProd]
                self.df_treemap = pd.DataFrame(data)
            else:
                data = [dict(source=prod.currProd, target=prod.nextProd)
                        for prod in ComDataInventory.orpDataInv.orpDepthInfoList
                        if prod.nextProd != prod.currProd if prod.dmdProdID in self.prodID]
                self.df_treemap = pd.DataFrame(data)

        self.df_treemap = self.df_treemap.drop_duplicates()

        temp = self.df_treemap['target'][~self.df_treemap['target'].isin(self.df_treemap['source'].unique())]
        new_df = pd.DataFrame(dict(source=temp, target=''))
        self.df_treemap = self.df_treemap.append(new_df, ignore_index=True)


    def generate_fig_treemap(self):
        self.fig_treemap = go.Figure(go.Treemap(labels=self.df_treemap['source'], parents=self.df_treemap['target']))

        self.fig_icicle = go.Figure(go.Icicle(labels=self.df_treemap['source'], parents=self.df_treemap['target']))

        self.fig_sunburst = go.Figure(go.Sunburst(labels=self.df_treemap['source'], parents=self.df_treemap['target']))

    def analysis_engine_result_lot_history(self, *args):
        """
        engine의 결과값을 그래프로 시각화 해줍니다.

        params
        'target'  : target별로 구분된 gantt chart를 출력합니다.
        'event'   : event별로 구분된 gantt chart를 출력합니다.
        'sankey'  : prod 변환 기록을 보여주는 sankey diagram을 출력합니다.
        'treemap' : prod 변환 기록을 보여주는 treemap charts를 출력합니다.
        """
        if 'event' in args or 'target' in args:
            self.generate_dataframe()
            self.append_track_in_df()

        if 'target' in args:
            self.generate_fig_target_col()
            self.fig_target_col.write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
            self.fig_target_col.show(width=2000, height=750)

        if 'event' in args:
            self.generate_fig_event_col()
            self.fig_event_col.write_html(f"{PathInfo.xlsx}{os.sep}temp_event.html", default_width=2300, default_height=900)
            self.fig_event_col.show(width=2000, height=750)

        if 'sankey' in args:
            if 'treemap' in args:
                self.generate_sankey_dataframe(flag_treemap=True)
                self.generate_fig_sankey_diagram()
                self.fig_sankey.write_html(f"{PathInfo.xlsx}{os.sep}temp_sankey.html")
                self.fig_sankey.write_image(f"{PathInfo.xlsx}{os.sep}temp_sankey.png")
                self.fig_sankey.show()
            else:
                self.generate_sankey_dataframe(flag_treemap=False)
                self.generate_fig_sankey_diagram()
                self.fig_sankey.write_html(f"{PathInfo.xlsx}{os.sep}temp_sankey.html")
                self.fig_sankey.write_image(f"{PathInfo.xlsx}{os.sep}temp_sankey.png")
                self.fig_sankey.show()

        if 'treemap' in args:
            if 'sankey' in args:
                self.generate_treemap_dataframe(flag_sankey=True)
                self.generate_fig_treemap()
                self.fig_treemap.write_html(f"{PathInfo.xlsx}{os.sep}temp_treemap.html")
                self.fig_treemap.show()
            else:
                self.generate_treemap_dataframe(flag_sankey=False)
                self.generate_fig_treemap()
                self.fig_treemap.write_html(f"{PathInfo.xlsx}{os.sep}temp_treemap.html")
                self.fig_treemap.show()
                self.fig_icicle.show()
                self.fig_sunburst.show()
