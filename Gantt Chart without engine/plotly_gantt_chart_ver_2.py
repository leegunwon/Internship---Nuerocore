import plotly.figure_factory as ff
import pandas as pd
import random
import datetime
import string
import plotly.express as px
import time

class Generator:
    def __init__(self, lots):
        self.lots = lots
        self.start_list = []
        self.finish_list = []
        self.resource_list = []
        self.lot_list = []
    def lot_generator(self):
        for i in range(self.lots):
            j = str(i)
            self.lot_list.append("Lot" + j)

    def date_generator(self, year, month, day):
        for i in range(self.lots):
            self.start_list.append(str(datetime.date(year, month, day) + datetime.timedelta(days=i)))
        for i in range(self.lots):
            self.finish_list.append(str(datetime.date(year, month, day) + datetime.timedelta(days=i+1)))

    def resource_generator(self):
        for j in range(self.lots):
            rand_str = ""
            for i in range(5):
                rand_str += str(random.choice(string.ascii_uppercase))
            self.resource_list.append(rand_str)

    def dataframe_generator(self):
        dataframe_list = []
        for j in range(len(self.lot_list)):
            dataframe_list.append(dict(Task=self.lot_list[j], Start=self.start_list[j],
                                            Finish=self.finish_list[j], Resource=self.resource_list[j]))
        self.df =pd.DataFrame(dataframe_list)

    
    def fig_generator(self):
        self.fig = ff.create_gantt(self.df, show_colorbar=True, group_tasks=True, index_col='Complete', colors = 'Viridis')

if __name__ == "__main__":
    test1 = Generator(10)
    test1.lot_generator()
    test1.lot_generator()
    test1.date_generator(2010, 12, 28)
    test1.date_generator(2011, 1, 7)
    test1.resource_generator()
    test1.resource_generator()
    test1.dataframe_generator()
    test1.df['Complete'] = [10,20,30,40,50,60,70,80,90,100,100,90,20,40,60,50,40,50,20,10]
    test1.fig_generator()
    test1.fig.show()
    test1.fig_px.show()