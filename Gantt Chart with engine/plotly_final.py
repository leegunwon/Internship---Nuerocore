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
import numpy as np
import time
import os


def start(dataset_id: str, simulation_prefix: str):
    # Parameters.set_plan_horizon(1)
    # Parameters.set_engine_database()
    Parameters.operation_dsp_rules = {"2100": "ORP_RES_REQ"}
    SimulatorConfigurator.configurate_simulator(dataset_id=dataset_id, simulation_prefix=simulation_prefix)
    SimulatorConfigurator.run_simulator()
    # whID: warehouse id, currProd: PS.FROM_PROD_CODE, nextProd: PS.TO_PROD_CODE

    try1 = Analysis(prod_id=[], lot_id=[])
    try1.analysis_engine_result_lot_history('target', 'event', 'sankey', 'treemap')




class Analysis:

    def __init__(self, **kwargs):
        self.prodID = kwargs.get('prod_id', [])
        self.id_list = kwargs.get('lot_id', list(SiteModels.lots.keys())) or []


        # target을 기준으로 color를 나눈 figure입니다
        self.fig_target_col = None
        # event를 기준으로 color를 나눈 figure
        self.fig_event_col = None
        # sankey diagram의 figure
        self.fig_sankey = None
        # treemap charts의 figure
        self.fig_treemap = None
        # icicle charts의 figure
        self.fig_icicle = None
        # sunburst charts의 figure
        self.fig_sunburst = None

        # 메인 데이터프레임입니다.
        self.df = None
        # late표시를 하기 위한 데이터프레임
        self.df_late = None
        # sankey의 dataframe
        self.df_sankey = self.df_sankey = pd.DataFrame([])
        # treemap, icicle charts, sunburst charts의 dataframe
        self.df_treemap = self.df_sankey = pd.DataFrame([])

        # sankey의 prodID를 숫자로 치환하기 위해 만든 리스트 형태의 prodID 카테고리
        self.label_name = None
        # lot_id별 모든 공정이 끝났을 때의 시간
        self.end_time = {}
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
        lot history를 리스트 컴프리헨션으로 만들 수 있지만 end_time과 facility_utilization 연산을 해야 하고 데이터 프레임 두개를 생성해야 함으로
        한 개의 for문으로 구성하였습니다. (리스트 컴프리헨션으로 작성 시 소요시간이 증가했습니다.)

        facility_utiliztion_rate과 end_time time이 생성됩니다.

        facility_utilization = (track_in event 수행 시간 / 전체 수행 시간) * 100
        end_time = 각 lot들의 event가 모두 끝나는 시간입니다.


        df(데이터프레임)를 생성합니다.

        self.df_late = 지연됐음을 표현하기 위한 검정 바를 생성하기 위한 데이터 프레임입니다.
                        (target과 event의 정보가 의미가 없음으로 target = '', event = 'late'로 표현 target과 event 명은 바꾸어도 상관없습니다.)

        self.df = lot_history의 정보를 담고 있는 데이터 프레임입니다.

        df의 column(lot_id, target, event, start, operation_time, lpst)

        lot_id = 어떤 lot의 hisotry인지
        target = 어떤 머신이 공정을 수행하는지
        event = 어떤 공정이 발생하였는지
        start = 언제 시작되었는지
        operation_time = 얼마나 시간이 소모되었는지
                        (figure 생성에 사용되는 초 단위는 밀리초 history에 저장된 단위는 초 이므로 *1000을 곱합니다.)
        lpst = 납기 기한을 맞추려면 최소한 언제 시작되어야 하는지
        """

        if self.id_list == []:
            self.id_list = SiteModels.lots.keys()
        lot_history_data = []
        late_lot_history = []

        # append 함수 호출 시간을 단축시키기 위해 변수에 저장하여 사용
        lot_history_data_app = lot_history_data.append
        late_lot_history_app = late_lot_history.append

        for lot_ids in self.id_list:
            histories = SiteModels.lots[lot_ids].history
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
                lot_history_data_app(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                          start=start_datetime, operation_time=(history[3] - history[2]) * 1000,
                                          lpst=lpst_datetime))

                if history[1] == "TRACK_IN":
                    sum_track_in_time += history[3] - history[2]

                # TRACK_IN event의 지연됨 표현은 하지 않기 때문에 elif를 사용합니다.
                # lpst보다 start가 더 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif lpst_datetime < start_datetime:
                    late_lot_history_app(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                              start=start_datetime, operation_time=(history[3] - history[2]) * 1000,
                                              lpst=lpst_datetime))
                # lpst보다 start가 더 빠르지만 finish보다 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif start_datetime <= lpst_datetime <= finish_datetime:
                    late_lot_history_app(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                              start=lpst_datetime, operation_time=(history[3] - history[5]) * 1000,
                                              lpst=lpst_datetime))

            # lot_id를 key, lot_id별 가장 늦은 시간을 value로 하는 딕셔너리 생성
            self.end_time[lot_ids] = finish_d

            # facility_utilization = (track_in event 수행 시간 / 전체 수행 시간) * 100
            if finish_d - start_d != 0:
                self.facility_utilization_rate[lot_ids] = sum_track_in_time * 100 / (finish_d - start_d)
            else:
                self.facility_utilization_rate[lot_ids] = 0

        # 데이터 프레임 2개 생성 self.df = lot_history, self. df_late = 지연됐음을 표현하기 위한 검정 바
        self.df = pd.DataFrame(lot_history_data)
        self.df_late = pd.DataFrame(late_lot_history)

        # track_in event를 df(데이터프레임) 가장 밑 행으로 배치
        self.to_bottom_track_in_df()

    def to_bottom_track_in_df(self):
        """

        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시켜서 차트가 가장 마지막에 생성되도록 합니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.

        """
        track_in_df = self.df.loc[self.df['event'] == 'TRACK_IN']
        self.df = self.df[self.df['event'] != 'TRACK_IN']
        self.df = self.df.append(track_in_df, ignore_index=True)

    def update_layout_target(self):
        """

        figure(fig_target_col)의 [layout] 항목을 수정합니다.

        1. layout.annotations (주석)

        x = x축 좌표 (현재 모든 공정이 끝난 시간에 20000초를 더한 값으로 설정했습니다.)
        y = y축좌표 (현재 해당하는 lot_id로 설정했습니다.)
        text = 표시할 텍스트를 지정합니다. (lot_id별 facility_utilization 값)
        showarrow = annotation에 화살표를 표시할 것 인지 결정합니다. (default가 True라 False로 지정해줘야 합니다.)

        2. barmode: 그래프 바의 mode를 설정 합니다.
        (mode 종류)
        stack: 바가 쌓아 지는 구조입니다. (그래프 바의 값만큼 더해져가는 구조입니다.)
        relative: 음수는 음수 쪽으로 쌓이고 양수는 양수 쪽으로 쌓이는 stack 구조입니다.
        group: 동일한 값에 여러 개의 바가 한개의 그룹으로 묶여서 있는 구조입니다.( ex) 한 개의 lot_id y축 좌표에 두 개의 바 생성하고 싶을 때 사용 가능합니다.)
        overlay: 다른 모드와는 다르게 시작 지점을 원하는 곳으로 지정 가능합니다. (간트차트의 경우 이 방법이 가장 적합합니다.)

        3. xaxis.type: x축의 타입을 지정합니다.
        종류   [‘-‘, ‘linear’, ‘log’, ‘date’, ‘category’]

        """
        # facility utilization rate를 주석으로 설정했습니다.
        anno_facility_util = [dict(x=Analysis.str_to_datetime(self.end_time[lot_id] + 20000),
                                   y=lot_id,
                                   text=f"{self.facility_utilization_rate[lot_id]:.1f}%",
                                   showarrow=False) for lot_id in self.id_list]

        # Update layout with annotations and bar mode
        self.fig_target_col.update_layout(
            annotations=anno_facility_util,
            # 바가 생성되는 방식을 설정했습니다..
            barmode='overlay',
            # x축 눈금의 타입을 설정합니다.
            xaxis_type='date',
            title_text="gantt chart target"
        )

    def generate_fig_target_col(self):
        """

        target을 기준으로 color를 나눈 figure를 작성합니다.

        figure
        1. data (trace)
        2. layout
        3. frame

        trace는 figure 생성시에 그래프 개체들의 정보 [data](한 개의 figure안에 여러 개의 trace로 구성 가능)

        trace 생성 param
        base            시작 시간
        x               소요된 시간
        y               y축 변수
        orientation     막대의 방향 (세로= default , 가로=h)
        legendgroup     레전드의 이름
        name            막대 그래프 개체 각각의 이름  (name에 설정된 이름으로 legend 이름이 설정됩니다.)
        text            그래프 내부에 표시 할 데이터
        hovertemplate   호버에 데이터를 표시할 형식
        hoverlabel      호버 label의 layout을 수정합니다. (hoverlabel.font: family(폰트), color, size를 지정할 수 있음)
        customdata      호버에 표시할 데이터입니다. (interactive한 기능을 사용할 때 주로 사용됨
                                                현재 연관된 기능을 사용하고 있지 않아 데이터 저장소로 사용)

        """
        #
        self.fig_target_col = go.Figure()

        # target별로 구분하여 legend를 생성하고 color를 지정하기 위해 target별 figure 생성
        for targ in self.df.target.unique():
            dff = self.df.loc[self.df.target == targ]
            self.fig_target_col.add_trace(
                go.Bar(base=dff.start, x=dff.operation_time, y=dff.lot_id, orientation='h', hoverlabel=dict(font=dict(size=15)),
                       legendgroup=targ, name=targ, text=targ, textposition='inside', customdata=dff.event,
                       hovertemplate='lot_id=%{y}<br>start=%{base}<br>finish=%{x}<br>target=' + targ + '<br><extra></extra>'))

        # self.df_late가 빈 데이터프레임일 경우 figure 생성 에러가 발생 하므로
        # 에러를 회피하기 위해 데이터프레임 shape로 빈 데이터프레임인지 판별
        if self.df_late.shape[0] != 0:
            self.fig_target_col.add_trace(
                go.Bar(base=self.df_late.start, x=self.df_late.operation_time, y=self.df_late.lot_id, orientation='h',
                       legendgroup='late', name='late', opacity=0.6, marker=dict(color='black'), hoverlabel=dict(font=dict(size=15)),
                       customdata=self.df_late.lpst, textposition='inside',
                       hovertemplate='lot_id=%{y}<br>start=%{base}<br>finish=%{x}<br>lpst=%{customdata}<extra></extra>'))

        # TRACK_IN event의 너비, 투명도, 텍스트 위치를 조정
        [(Analysis.modify_opacity(bar, 0.8),
          Analysis.modify_text_pos(bar, 'outside'),
          Analysis.modify_width(bar, 0.5))
         for bar in self.fig_target_col.data
         if ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]

        self.update_layout_target()

    def update_layout_event(self):
        """
        1. layout.annotations (주석) [anno_fact_util]
            x: x축 좌표는 현재 모든 공정이 끝난 시간에 20000초를 더한 값으로 설정합니다.
            y: y축 좌표는 현재 lot_id에 해당하는 좌표 값으로 설정합니다.
            text: 표시할 텍스트는 lot_id별 facility_utilization 값입니다.
            showarrow: annotation에 화살표를 표시할 것인지 결정합니다. 기본값은 True이므로 False로 지정해야 합니다.

        2. barmode: 그래프 바의 mode를 설정합니다. [self.fig_event_col.update_layout(barmode)]
            mode 종류:
              stack: 바가 쌓아 지는 구조입니다. 음수가 나오면 그 값에서 음의 방향으로 쌓입니다.
              relative: 음수는 음수 쪽으로 쌓이고 양수는 양수 쪽으로 쌓이는 stack 구조입니다.
              group: 동일한 값에 여러 개의 바가 한 개의 그룹으로 묶여서 있는 구조입니다.
              overlay: 다른 모드와는 다르게 시작 지점을 원하는 곳으로 지정할 수 있습니다.

        3. xaxis.type: x축의 타입을 지정합니다. 종류는 ['-','linear','log','date','category'] 중에서 선택할 수 있습니다.

        """
        # annotations
        anno_facility_util = [dict(x=Analysis.str_to_datetime(self.end_time[lot_id] + 20000),
                                  y=lot_id,
                                  text=f"{self.facility_utilization_rate[lot_id]:.1f}%",
                                  showarrow=False) for lot_id in self.id_list]
        # barmode, xaxis, annotations
        self.fig_event_col.update_layout(
            annotations=anno_facility_util,
            barmode='overlay',
            xaxis_type='date',
            title_text="gantt chart event"
        )

        # bar properties
        for bar in self.fig_event_col.data:
            if 'WH' in bar.legendgroup:
                # WH의 색상을 바꾸고 싶을 때
                Analysis.modify_color(bar, '#7FB3D5')
                Analysis.modify_legendgroup(bar, 'WH')
            # TRACK_IN event의 막대가 잘 구분이 안될 때
            elif 'TRACK_IN' in bar.legendgroup:
                # TRACK_IN event의 막대 두께
                Analysis.modify_width(bar, 0.5)
                # TRACK_IN event의 막대 투명도
                Analysis.modify_opacity(bar, 0.8)
                # TRACK_IN event의 텍스트 위치
                Analysis.modify_text_pos(bar, 'outside')

    def generate_fig_event_col(self):
        """

        event를 기준으로 color를 나눈 figure를 작성합니다.

        figure  (보통 frame은 조작하지 않음)
        1. data,
        2. layout
        3. frame

        trace는 figure 생성시에 그래프 개체들의 정보 [data](한 개의 figure안에 여러 개의 trace로 구성 가능)

        trace 생성 param
        base            시작 시간
        x               소요된 시간
        y               y축 변수
        orientation     막대의 방향 (세로= default , 가로=h)
        legendgroup     레전드의 이름
        opacity         그래프 개체 투명도
        marker          그래프 개체 세부 정보 (marker.color 그래프 개체 색상)
        name            막대 그래프 개체 각각의 이름
        text            그래프 내부에 표시 할 데이터
        hovertemplate   호버에 데이터를 표시할 형식 (<br> 줄이 바뀝니다.
                                                <extra>{fullData.name}</extra> dullData.name에 해당하는 데이터가 보조 상자에 출력됩니다.
                                                <extra></extra>로 작성시 보조 상자가 출력되지 않습니다.
                                                %{variable} figure에 variable로 저장된 데이터가 출력됩니다.
                                                ex) %{base}로 작성했을 때 hover를 표시하는 객체의 base 값이 출력됩니다.
                                                )
        hoverlabel      호버 label의 layout을 수정합니다. (hoverlabel.font: family(폰트), color, size를 지정할 수 있음)
        customdata      호버에 표시할 데이터 (interactive한 기능을 사용할 때 주로 사용됨
                                          현재 연관된 기능을 사용하고 있지 않아 데이터 저장소로 사용)

        """

        # 그래프의 색상 리스트입니다.

        col_list = ['blue', '#00CFFF', '#FCA40B', 'limegreen', '#1f77b4', '#F5A9D0']
        colors = iter(col_list)

        self.fig_event_col = go.Figure()

        for eve in self.df.event.unique():
            dff = self.df.loc[self.df.event == eve]

            self.fig_event_col.add_trace(
                go.Bar(
                    # x 시작점 값입니다.
                    base=dff.start,
                    # base에 설정된 시작점에서 x에 설정된 값만큼 바가 생성됩니다.
                    x=dff.operation_time,
                    y=dff.lot_id,
                    # default 값은 세로 막대 바로 생성됩니다. 'h'로 변경 시 가로 막대 바로 생성됩니다.
                    orientation='h',
                    # 마우스 커서를 대면 나오는 텍스트 박스의 크기 바꾸고 싶을 때
                    hoverlabel=dict(font=dict(size=20)),
                    # 막대안에 나타나는 텍스트 내용 바꾸고 싶을 때
                    text=dff.target,
                    # legend에 표시되는 이름입니다.
                    name=eve,
                    # 동일한 이름으로 설정하면 legend를 한개의 그룹으로 묶어서 동작할 수 있게 해줍니다.
                    legendgroup=eve,
                    # 막대의 색상 바꾸고 싶을 때
                    marker=dict(color=next(colors), line=dict(width=1, color='black')),
                    # 텍스트 위치를 조정합니다.['inside', 'outside']
                    textposition='inside',
                    # hover 텍스트로 불러오기 위해 customdata에 임시 저장  %{customdata}로 불러옴
                    customdata=dff.target,
                    # 마우스 커서를 가져다 대면 나오는 hover의 텍스트 내용의 형식을 지정할 수 있습니다.(html형식으로 지정 가능합니다.)
                    hovertemplate=(
                            'lot_id=%{y}<br>'
                            'start=%{base}<br>'
                            'finish=%{x}<br>'
                            'target=%{customdata}<br>'
                            'event=' + eve + '<extra>%{x}</extra>'
                    )
                )
            )

        # lpst보다 늦은 부분을 검은색 처리를 하기 위해 추가하는 trace
        self.fig_event_col.add_trace(
            go.Bar(
                # x 시작점 값입니다.
                base=self.df_late.start,
                # base에 설정된 시작점에서 x에 설정된 값만큼 바가 생성됩니다.
                x=self.df_late.operation_time,
                y=self.df_late.lot_id,
                # default = 세로 막대 바로 생성됩니다. 'h' = 가로 막대 바로 생성됩니다.
                orientation='h',
                # 동일한 이름으로 설정하면 legend를 한개의 그룹으로 묶어서 동작할 수 있게 해줍니다.
                legendgroup='late',
                # legend에 표시되는 이름입니다.
                name='late',
                opacity=0.6,
                # 막대의 색상 바꿀 수 있습니다.
                marker=dict(color='black'),
                # hover 텍스트로 불러오기 위해 customdata에 임시 저장
                # hovertemplate에서 %{customdata}로 설정하면 hover 텍스트로 생성할 수 있습니다.
                customdata=self.df_late.lpst,
                # 텍스트 위치를 조정합니다.['inside', 'outside']
                textposition='inside',
                # 마우스 커서를 대면 나오는 텍스트 박스(hover)의 크기를 바꿉니다.
                hoverlabel=dict(font=dict(size=20)),
                # 마우스 커서를 가져다 대면 나오는 hover의 텍스트 내용 바꾸고 싶을 때
                hovertemplate=(
                    'lot_id=%{y}<br>'
                    'start=%{base}<br>'
                    'finish=%{x}<br>'
                    'lpst=%{customdata}<extra></extra>'
                )
            )
        )

        self.update_layout_event()

    def generate_sankey_dataframe(self, flag_treemap: bool):
        """
        sankey_dataframe을 생성합니다.

        dataframe columns
        source = prod(현재)  'TO_PROD_CODE'
        target = prod(다음)  'FROM_PROD_CODE'
        value = 1  (value를 꼭 설정해 주어야 합니다.)

        prod name을 대응되는 숫자로 변경해 줘야 합니다.
        """

        # Get the necessary dataframes
        ps_df = ComDataInventory.input_data.data_frames['in_MainPsData']
        df_prod = PandasUtil.select(ps_df, ['FROM_PROD_CODE', 'TO_PROD_CODE'])

        # If no prodID is specified, add the entire dataframe to self.df_sankey
        if not self.prodID:
            self.df_sankey = df_prod.copy()
        else:
            # prodID까지의 라우팅을 찾기 위한 과정입니다.
            while True:
                temp_df = df_prod.loc[df_prod.TO_PROD_CODE.isin(self.prodID)]
                self.df_sankey = self.df_sankey.append(temp_df)
                self.prodID = temp_df['FROM_PROD_CODE']

                if temp_df.empty:
                    break

        # 트리맵을 생성한다면 prod name을 숫자로 바꾸기 전에 df_treemap에 계승합니다.
        if flag_treemap:
            self.df_treemap = self.df_sankey.copy()

        # Rename columns and set value to 1
        self.df_sankey.columns = ['source', 'target']
        self.df_sankey['value'] = 1

        # Sort the dataframe by source column
        self.df_sankey.sort_values(by='source')

        # Create the prod category (self.label_name)
        source_list = self.df_sankey.source.unique()
        targ_list = self.df_sankey.target.unique()
        self.label_name = list(set(np.append(source_list, targ_list)))

        # Convert prod names to numbers
        for a, b in enumerate(self.label_name):
            self.df_sankey = self.df_sankey.replace(b, a)

    def generate_fig_sankey_diagram(self):
        """

        sankey_diagram의 figure를 작성합니다.

         figure 구성  (보통 frame은 조작하지 않음)
        1. data,
        2. layout
        3. frame

        data 생성 param
        node(sankey의 상자 부분)
        label 각 노드 상자의 이름
        line 노드 상자의 테두리

        link(각 노드를 연결하고 있는 부분)
        source 두 개의 노드를 연결할 때 시작점 (숫자 형태의 데이터로 저장되고 그 값은 node의 label값과 매칭됩니다.)
        target 두 개의 노드를 연결할 때 끝점   ( ex) target=[1]의 의미는 target=[label[1]]입니다.)
        value  몇 개가 source에서 target으로 변화했는지 나타내는 부분
        color  노드를 연결하는 부분의 색상

        layout
        title_text = 그래프 좌측 상단에 표시되는 title의 text 내용입니다.

        """
        # Define nodes
        nodes = dict(
            label=self.label_name,
            line=dict(color="black"),
            pad=15,
            thickness=20
        )

        # Add links
        links = dict(
            source=self.df_sankey.source,
            target=self.df_sankey.target,
            value=self.df_sankey.value,
            # 노드들을 연결하는 링크의 색을 설정합니다.
            color='grey'
        )

        self.fig_sankey = go.Figure(data=[go.Sankey(node=nodes, link=links)])


        self.fig_sankey.update_layout(
            title_text="sankey diagram"
        )

    def generate_treemap_dataframe(self, flag_sankey: bool):
        """
        treemap_dataframe을 생성합니다.
        treemap_dataframe의 경우 sankey_dataframe과 유사하므로
        sankey_dataframe이 생성되었을 경우 계승하여 사용 하였습니다.

        dataframe columns
        source = prod(변하기 전)
        target = prod(변한 후)
        value는 작성해도 안해도 관계없습니다.

        sankey처럼 prod를 숫자의 형태로 변환하지 않습니다.
        sankey의 경우 중복이 허용되지만 treemap의 경우 중복이 허용되지 않습니다.

        """

        ps_df = ComDataInventory.input_data.data_frames['in_MainPsData']
        df_prod = PandasUtil.select(ps_df, ['FROM_PROD_CODE', 'TO_PROD_CODE'])

        # sankey를 출력하고 있지 않다면 treemap_df를 만들고, sankey를 출력하고 있다면 df를 계승하여 사용하기 위해 구분합니다.
        if not flag_sankey:
            if not self.prodID:
                self.df_treemap = df_prod.copy()
            else:
                # Traverse up the product hierarchy for each prodID
                while True:
                    temp_df = df_prod.loc[df_prod.TO_PROD_CODE.isin(self.prodID)]
                    self.df_treemap = self.df_treemap.append(temp_df)
                    self.prodID = temp_df['FROM_PROD_CODE']

                    if temp_df.shape[0] == 0:
                        break
        else:
            pass

        self.df_treemap.columns = ['source', 'target']
        temp = self.df_treemap['target'][~self.df_treemap['target'].isin(self.df_treemap['source'].unique())]
        new_df = pd.DataFrame(dict(source=temp, target=' '))
        self.df_treemap = self.df_treemap.append(new_df, ignore_index=True).drop_duplicates()

    def generate_fig_treemap(self):
        """
        treemap figure를 생성합니다.

        params
        labels = prod(현재)  == df_treemap['source']
        parents = prod(다음)  == df_treemap['target']
        """

        self.fig_treemap = go.Figure(go.Treemap(labels=self.df_treemap['source'], parents=self.df_treemap['target'], maxdepth=3))

    def analysis_engine_result_lot_history(self, *args):
        """
        engine의 결과값을 그래프로 시각화 해줍니다.

        params
        'target'  : target별로 구분된 gantt chart를 출력합니다.
        'event'   : event별로 구분된 gantt chart를 출력합니다.
        'sankey'  : prod 변환 기록을 보여주는 sankey diagram을 출력합니다.
        'treemap' : prod 변환 기록을 보여주는 treemap charts를 출력합니다.

        'server_data -> xlsx'에  temp_(charts명)의 이름의 html 파일로 저장됩니다.

        write_html = charts를 html 형식의 파일로 저장합니다.
        write_image = charts를 png 형식의 파일로 저장합니다.
        show = charts를 출력합니다.
        (width = charts의 가로 , height = charts의 세로)
        """

        if 'event' in args or 'target' in args:
            self.generate_dataframe()

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
                self.fig_sankey.show()
            else:
                self.generate_sankey_dataframe(flag_treemap=False)
                self.generate_fig_sankey_diagram()
                self.fig_sankey.write_html(f"{PathInfo.xlsx}{os.sep}temp_sankey.html")
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
