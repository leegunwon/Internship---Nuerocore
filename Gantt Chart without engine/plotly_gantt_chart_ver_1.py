import plotly.express as px
import pandas as pd
import random
import datetime
import string

class Generator:
    def __init__(self, lots):
        self.lots = lots

    def lot_generator(self, lots):
        self.lot_list = []
        for i in range(lots):
            j = str(i)
            self.lot_list.append("Lot" + j)

    def date_generator(self, start):
        self.start_list = []
        self.finish_list = []
        year = int(start[0:4])
        month = int(start[5:7])
        date = int(start[8:10])
        for i in range(self.lots):
            self.start_list.append(str(datetime.date(year, month, date) + datetime.timedelta(days=i)))
        for i in range(self.lots):
            self.finish_list.append(str(datetime.date(year, month, date) + datetime.timedelta(days=i+1)))

    def resource_generator(self):
        self.resource_list = []
        for j in range(self.lots):
            rand_str = ""
            for i in range(5):
                rand_str += str(random.choice(string.ascii_uppercase))
            self.resource_list.append(rand_str)

    def dataframe_generator(self):
        self.dataframe_list = []
        for i in range(len(self.lot_list)):
            temp_data = [self.lot_list[i], self.start_list[i], self.finish_list[i], self.resource_list[i]]
            self.dataframe_list.append(temp_data)
    def datatype_translator(self):
        self.dataf =pd.DataFrame(self.dataframe_list,
                  index = (range(len(self.dataframe_list))),
                  columns=('lot','Start', 'Finish', 'machine'))
    def fig_generator(self):
        self.fig = px.timeline(self.dataf, x_start='Start', x_end="Finish", y="lot", color="machine")

    def simulation(self):
        self.lot_generator(self.lots)
        self.date_generator('2009-01-01')
        self.resource_generator()
        self.dataframe_generator()
        self.datatype_translator()
        self.fig_generator()
        self.fig.show()
exam = Generator(10)
exam.simulation()
exam1 = Generator(10)
exam1.simulation()