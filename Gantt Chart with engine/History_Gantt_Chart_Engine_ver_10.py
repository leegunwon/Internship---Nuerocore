
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import time
import os
import numpy as np

class Analysis:

    def __init__(self, **kwargs: list):
        # args에 입력된 lot_id에 해당하는 lot들의 history만 출력합니다.
        # args를 입력하지 않으면 모든 lot들이 출력됩니다.
        if 'prodID' in kwargs.keys():
            if kwargs['prodID'] != None:
                self.prodID = [prodID for prodID in kwargs['prodID']]
            else: self.prodID=[]
        else: self.prodID=[]

        if 'id_list' in kwargs.keys():
            if kwargs['id_list'] != None and kwargs['id_list'] != []:
                self.id_list = [lot_id for lot_id in kwargs['id_list']]
            else:
                self.id_list = [lot_id for lot_id in SiteModels.lots.keys()]
        else:
            self.id_list = [lot_id for lot_id in SiteModels.lots.keys()]

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
    def modify_text_pos(bar, pos):
        bar.textposition = pos

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
                start_datetime = Analysis.str_to_datetime(history[2])
                finish_datetime = Analysis.str_to_datetime(history[3])
                lpst_datetime = Analysis.str_to_datetime(history[5])


                # 그래프로 표현할 lot의 history 결과를 출력합니다.
                lot_history_data.append(dict(lot_id=lot_ids, target=history[0], event=history[1],
                                             start=start_datetime, finish=finish_datetime, operation_time=(history[3] - history[2])*1000, lpst=lpst_datetime))

                if history[1] == "TRACK_IN":
                    sum_track_in_time += history[3] - history[2]

                # TRACK_IN event의 지연됨 표현은 하지 않기 때문에 elif를 사용합니다.
                # lpst보다 start가 더 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif lpst_datetime < start_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                     start=start_datetime,  operation_time=(history[3] - history[2])*1000, lpst=lpst_datetime))

                # lpst보다 start가 더 빠르지만 finish보다 느릴 때 지연됨을 표현하기 위해 late_lot_history에 정보를 저장합니다.
                elif start_datetime <= lpst_datetime <= finish_datetime:
                    late_lot_history.append(dict(lot_id=lot_ids, target=' ', event='late',
                                                     start=lpst_datetime, operation_time=(history[3] - history[5])*1000, lpst=lpst_datetime))

            # lot_id를 key, lot_id별 가장 늦은 시간을 value로 하는 딕셔너리 생성
            self.latest_finish[lot_ids] = finish_d

            # lot_id를 key, lot_id별 가장
            if finish_d - start_d !=0:
                self.facility_utiliztion_rate[lot_ids] = sum_track_in_time * 100 / (finish_d - start_d)
            else:
                self.facility_utiliztion_rate[lot_ids] = 0

        # 지연된 정보를
        self.df = pd.DataFrame(lot_history_data)
        self.df1 = pd.DataFrame(late_lot_history)

    def update_layout_target(self):
        """
        figure(fig_target_col)의 layout 항목을 수정합니다.
        """
        anno_factory_utilization = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 20000), y=lot_id,
                               text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%", showarrow=False)
                          for lot_id in self.id_list]

        # self.fig_target_col.layout.hovermode = "x"    # hovermode입니다 x축 기준 동일 선상에 있는 모든 hover가 출력됩니다.
        with self.fig_target_col.batch_update():
            self.fig_target_col.layout["annotations"] = anno_factory_utilization
            self.fig_target_col.update_layout(xaxis=dict(type='date'), barmode='overlay')
            # TRACK_IN event의 너비와 투명도를 조정하였습니다.
            [(Analysis.modify_opacity(bar, 0.8), Analysis.modify_text_pos(bar, 'outside'), Analysis.modify_width(bar, 0.5)) for bar in self.fig_target_col.data if
             ('CM' in bar.legendgroup or 'LM' in bar.legendgroup)]

    def generate_fig_target_col(self):
        """
        target을 기준으로 color를 나눈 figure를 작성합니다.
        """
        self.fig_target_col = go.Figure()
        with self.fig_target_col.batch_update():
            for targ in self.df.target.unique():
                dff = self.df.loc[self.df.target == targ]
                self.fig_target_col.add_trace(go.Bar(base=dff.start, x=dff.operation_time, y=dff.lot_id, orientation='h',
                                     xaxis='x', yaxis='y', name=targ, legendgroup=targ, text=targ, hovertemplate='<b>lot_id=%{y}<br>target='+targ+'<br>start=%{base}<br>finish=%{x}<extra></extra>'))

            self.fig_target_col.add_trace(go.Bar(base=self.df1.start, x=self.df1.operation_time, y=self.df1.lot_id, orientation='h',
                                                 name='late', legendgroup='late', opacity=0.6, marker=dict(color='black'), hovertext=self.df1.lpst))
        self.fig_target_col.update_traces(textposition='inside')
        self.update_layout_target()

    def update_layout_event(self):
        """
        figure(fig_target_col)의 layout 항목을 수정합니다.
        """
        anno_fact_util = [dict(x=Analysis.str_to_datetime(self.latest_finish[lot_id] + 20000), y=lot_id,
                               text=f"{self.facility_utiliztion_rate[lot_id]:.1f}%", showarrow=False) for lot_id in self.id_list]

        with self.fig_event_col.batch_update():
            self.fig_event_col.layout["annotations"] = anno_fact_util
            self.fig_event_col.update_layout(xaxis=dict(type='date'), barmode='overlay')

            [(Analysis.modify_color(bar, '#7FB3D5'), Analysis.modify_legendgroup(bar, 'WH')) for bar in self.fig_event_col.data if
             ('WH' in bar.legendgroup)]

            [(Analysis.modify_width(bar, 0.5),Analysis.modify_opacity(bar, 0.7),Analysis.modify_text_pos(bar, 'outside')) for bar in self.fig_event_col.data if
             ('TRACK_IN' in bar.legendgroup)]

            [Analysis.modify_hov_name_from_cus_data(bar) for bar in self.fig_event_col.data if
             ('late' in bar.legendgroup)]


    def generate_fig_event_col(self):
        """
        event를 기준으로 color를 나눈 figure를 생성합니다.
        """
        col_list = ['#20B2AA', '#FAAC58', 'limegreen', '#F8C471' ,'forestgreen']
        colors = iter(col_list)

        self.fig_event_col = go.Figure()
        with self.fig_event_col.batch_update():
            for eve in self.df.event.unique():
                dff = self.df.loc[self.df.event == eve]
                self.fig_event_col.add_trace(go.Bar(base=dff.start, x=dff.operation_time, y=dff.lot_id, orientation='h', text=dff.target,
                                                    name=eve, legendgroup=eve, marker=dict(color=next(colors)), hovertemplate='<b>lot_id=%{y}<br>event='+eve+'<br>start=%{base}<br>finish=%{x}<extra></extra>'))
            self.fig_event_col.add_trace(
                go.Bar(base=self.df1.start, x=self.df1.operation_time, y=self.df1.lot_id, orientation='h',
                        name='late', legendgroup='late', opacity=0.6, marker=dict(color='black'), hovertext=self.df1.lpst))

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

    def generate_sankey_dataframe(self):
        """
        현재 prod명과 다음 prod명을 source와 target에 분류하고, prod개수를 1로 설정하여 데이터프레임을 작성.
        """
        if self.prodID==[]:
            data = [dict(source=prod.currProd, target=prod.nextProd, value=1) for prod in ComDataInventory.orpDataInv.orpDepthInfoList
                    if prod.nextProd != prod.currProd]
        else:
            data = [dict(source=prod.currProd, target=prod.nextProd, value=1) for prod in ComDataInventory.orpDataInv.orpDepthInfoList
                if prod.nextProd != prod.currProd if prod.dmdProdID in self.prodID]

        self.df_sankey_data = pd.DataFrame(data)

        source_list = self.df_sankey_data.source.unique()
        targ_list = self.df_sankey_data.target.unique()

        self.label_name = list(set(np.append(source_list, targ_list)))
        self.label_name.sort()

        for i in range(len(self.label_name)):
            self.df_sankey_data = self.df_sankey_data.replace(self.label_name[i], i)

    def generate_fig_sankey_diagram(self):
        self.fig_sankey_diagram = go.Figure(data=[go.Sankey(
            # Define nodes
            arrangement='fixed',
            node=dict(
                label=self.label_name,
                line = dict(color = "black")
            ),

            # Add links
            link=dict(
                source=self.df_sankey_data.source,
                target=self.df_sankey_data.target,
                value=self.df_sankey_data.value,
                color='grey'
            ))])

        self.fig_sankey_diagram.update_layout(
            title_text="sankey diagram")

    def analysis_engine_result_lot_history(self, *args):
        """
        engine의 결과값을 그래프로 시각화 해줍니다.
        """
        self.generate_dataframe()
        # track_in 이벤트만 따로 모아놓은 데이터 프레임입니다.
        # self.append_track_in_df()

        if 'target' in args:
            self.generate_fig_target_col()
            self.fig_target_col.write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
            self.fig_target_col.show(width=2000, height=750)

        if 'event' in args:
            self.generate_fig_event_col()
            self.fig_event_col.write_html(f"{PathInfo.xlsx}{os.sep}temp_event.html", default_width=2300, default_height=900)
            self.fig_event_col.show(width=2000, height=750)

        if 'sankey' in args:
            self.generate_sankey_dataframe()
            self.generate_fig_sankey_diagram()
            self.fig_sankey_diagram.write_html(f"{PathInfo.xlsx}{os.sep}temp_sankey.html")
            self.fig_sankey_diagram.write_image(f"{PathInfo.xlsx}{os.sep}temp_sankey.png")
            self.fig_sankey_diagram.show()
