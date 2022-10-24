# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:05:50 2022

@author: Panitz
"""

import pickle
import yaml
import flixOptHelperFcts as helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # für Plots im Postprocessing
import matplotlib.dates as mdates

class cFlow_post():


    @property
    def color(self):
        # interaktiv, falls component explizit andere Farbe zugeordnet
        return self._getDefaultColor()        

    def __init__(self,aDescr,flixResults):
        self.label = aDescr['label']
        self.bus   = aDescr['bus']
        self.comp  = aDescr['comp']
        self.descr = aDescr
        self.flixResults = flixResults
        self.comp_post = flixResults.postObjOfStr(self.comp)
        # Richtung:    
        self.isInputInComp = aDescr['isInputInComp']    
        if self.isInputInComp:
            self.from_node = self.bus
            self.to_node = self.comp
        else:
            self.from_node = self.comp
            self.to_node = self.bus

      
    def extractResults(self, allResults):
        self.results = allResults[self.comp][self.label]
        self.results_struct = helpers.createStructFromDictInDict(self.results)
    def getFlowHours(self):
        flowHours = sum(self.results['val']* self.flixResults.dtInHours)
        return flowHours
      
    def getLoadFactor(self, small=1e-2):
        loadFactor = None
        if ('invest' in self.results.keys()) and ('nominal_val' in self.results['invest'].keys()):
            flowHours = self.getFlowHours()
            #  loadFactor = Arbeit / Nennleistung / Zeitbereich = kWh / kW_N / h 
            nominal_val = self.results['invest']['nominal_val']
            if nominal_val < small:
                loadFactor = None 
            else:
                loadFactor = flowHours / self.results['invest']['nominal_val'] / self.flixResults.dtInHours_tot
        return loadFactor
   
    def belongToStorage(self):
        if 'isStorage' in self.flixResults.infos_system['components'][self.comp].keys():
            return self.flixResults.infos_system['components'][self.comp]['isStorage']
        else:
            return False  
    def _getDefaultColor(self):
        return self.comp_post.color
        
    
class cCompOrBus_post():
    def __init__(self,label, aDescr, flixResults, color = None):
        self.label = label
        self.type  = aDescr['class']
        self.descr = aDescr
        self.flixResults = flixResults
        self.color = color
    
class flix_results():
    def __init__(self, nameOfCalc, results_folder = None, comp_colors = None): #,timestamp = None):
  
        self.label = nameOfCalc
        self.comp_colors = comp_colors
        # default value:
        if self.comp_colors == None:
            import plotly.express as px
            self.comp_colors = px.colors.qualitative.Light24
            # see: https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
            
        # 'z.B.' 2022-06-14_Sim1_gurobi_SolvingInfos
        filename_infos = 'results/' + nameOfCalc + '_solvingInfos.yaml'
        filename_data = 'results/' + nameOfCalc + '_data.pickle'
    
    
        with open(filename_infos,'rb') as f:
            self.infos = yaml.safe_load(f)
            self.infos_system = self.infos['system_description']
    
        with open(filename_data,'rb') as f:
            self.results = pickle.load(f)          
        self.results_struct = helpers.createStructFromDictInDict(self.results)
        
        # list of str:
        self.buses = self.__getBuses()
        self.comps = self.__getComponents()
        # list of post_obj:
        self.bus_posts = self.__getAllBuses()   
        self.comp_posts = self.__getAllComps()
        self.flows = self.__getAllFlows()
        
        # Zeiten:
        self.timeSeries = self.results['time']['timeSeries']
        self.timeSeriesWithEnd = self.results['time']['timeSeriesWithEnd']
        self.dtInHours = self.results['time']['dtInHours']
        self.dtInHours_tot = self.results['time']['dtInHours_tot']
      
    def __getBuses(self):
        try :
            return list(self.infos_system['buses'].keys())
        except:
            raise Exception('buses nicht gefunden')
      
    
    def __getComponents(self):
        try :
            return list(self.infos_system['components'].keys())
        except:
            raise Exception('components nicht gefunden')
      
    def __getAllFlows(self):
        flows = []
        flow_list = self.infos_system['flows']
        for flow_Descr in flow_list:      
            aFlow = cFlow_post(flow_Descr, self)
            aFlow.extractResults(self.results)
            flows.append(aFlow)    
        return flows

    
    def __getAllComps(self):
        comps = []
        comp_dict = self.infos_system['components']
        
        myColorIter = iter(self.comp_colors)
        for label,descr in comp_dict.items():      
            aComp = cCompOrBus_post(label, descr, self, color = next(myColorIter))
            comps.append(aComp)    
        return comps
    
    def __getAllBuses(self):
        buses = []
        bus_dict = self.infos_system['buses']
        for label,descr in bus_dict.items():      
            aBus = cCompOrBus_post(label, descr, self)
            buses.append(aBus)
        return buses
        
    
    def getFlowsOf(self, node, node2=None):
        inputs_node = []
        outputs_node = []
        
        if node not in (self.buses + self.comps):
            raise Exception('node \'' + str(node) + '\' not in buses or comps')
        if node2 not in (self.buses + self.comps) and (node2 is not None):
            raise Exception('node2 \'' + str(node2) + '\' not in buses or comps')
        
        for flow in self.flows:
            if node in [flow.bus, flow.comp] \
              and ((node2 is None) or (node2 in [flow.bus, flow.comp])):
                if node == flow.to_node:
                    inputs_node.append(flow)
                elif node == flow.from_node:
                    outputs_node.append(flow)
                else:
                    raise Exception('node ' + node + ' not in flow.from_node or flow.to_node' )
        
        return (inputs_node, outputs_node)
  
    def postObjOfStr(self, aStr):
        thePostObj = None
        for aPostObj in self.comp_posts + self.bus_posts:
            if aPostObj.label == aStr:
                thePostObj = aPostObj        
        return thePostObj
    
    
    @staticmethod
    # check if SeriesValues should be shown
    def isGreaterMinFlowHours(aFlowValues,dtInHours,minFlowHours):
        # absolute Summe, damit auch negative Werte gezählt werden:
        absFlowHours = sum(abs(aFlowValues * dtInHours))
        # print(absFlowHours)
        return absFlowHours > minFlowHours
      
  
    @staticmethod
    # für plot get values (as timeseries or sorted):    
    def __get_Values_As_DataFrame(flows, timeSeriesWithEnd ,dtInHours, minFlowHours, indexSeq = None):
        
        
        # Dataframe mit Inputs (+) und Outputs (-) erstellen:
        timeSeries = timeSeriesWithEnd[0:-1] # letzten Zeitschritt vorerst weglassen        
        y = pd.DataFrame() # letzten Zeitschritt vorerst weglassen
        y_color = []
        # Beachte: hier noch nicht als df-Index, damit sortierbar
        for aFlow in flows:        
            values = aFlow.results['val'] # 
            values[np.logical_and(values<0, values>-1e-5)] = 0 # negative Werte durch numerische Auflösung löschen 
            assert (values>=0).all(), 'Warning, Zeitreihen '+ aFlow.label_full +' in inputs enthalten neg. Werte -> Darstellung Graph nicht korrekt'
                                
            if flix_results.isGreaterMinFlowHours(values, dtInHours, minFlowHours): # nur wenn gewisse FlowHours-Sum überschritten
                y[aFlow.comp + '.' + aFlow.label] = + values # ! positiv!
                y_color.append(aFlow.color)

        def appendEndTimeStep(y, lastIndex):   
            # hänge noch einen Zeitschrtt mit gleichen Werten an (withEnd!) damit vollständige Darstellung
            lastRow = y.iloc[-1] # kopiere aktuell letzte

            lastRow = lastRow.rename(lastIndex) # Index ersetzen -> letzter Zeitschritt als index        
            y=y.append(lastRow) # anhängen
            return y
        
        # add index (for steps-graph)
        if indexSeq is not None: 
            y['dtInHours'] = dtInHours
            # sorting:
            y = y.loc[indexSeq]
            # index:
            indexWithEnd = np.append([0],np.cumsum(y['dtInHours'].values))
            del y['dtInHours']            
            y.index = indexWithEnd[:-1]
            lastIndex = indexWithEnd[-1]            
            
        else:
            # index:
            y.index=timeSeries
            lastIndex = timeSeriesWithEnd[-1] # withEnd        
        
        # add last step:
        y = appendEndTimeStep(y,lastIndex)
        return y, y_color
    
    def getLoadFactorOfComp(self,aComp):
        (in_flows, out_flows) = self.getFlowsOf(aComp)
        for aFlow in (in_flows+out_flows):
            loadFactor = aFlow.getLoadFactor()
            if loadFactor is not None:
                # Abbruch Schleife und return:
                return loadFactor 
        return None
           
    def getLoadFactorsOfComps(self,withoutNone = True):
        loadFactors = {}
        comps = self.__getComponents()
        for comp in comps:
            loadFactor = self.getLoadFactorOfComp(comp)
            if loadFactor is not None:
                loadFactors[comp] = loadFactor
        return loadFactors
      
    def getFlowHours(self,busOrComp,useInputs = True,skipZeros=True):
        small = 1e5
        FH = {}
        (in_flows, out_flows) = self.getFlowsOf(busOrComp)
        if useInputs:
            flows =in_flows 
        else:
            flows = out_flows
        for aFlow in flows:
            flowHours = aFlow.getFlowHours() 
            if flowHours > small:
                FH [aFlow.comp + '.' + aFlow.label] = aFlow.getFlowHours() 
        return FH
    
    
    # def plotFullLoadHours(self):
    #   FLH = self.getLoadFactor
      
      
        
    # def plotFullLoadHours(self):
    #   comps = self.__getComponents()
    #   for comp in comps:
    #     getFlowHours
    #     self.getFullLoadHoursOfComp(comp)
        
    
    def plotShares(self, busesOrComponents, useInputs=True, withoutStorage = True, minSum=.1, othersMax_rel=0.05, plotAsPlotly = False, title = None, unit = 'FlowHours'):      
        '''     
        Parameters
        ----------
        busesOrComponents : str or list of str
            DESCRIPTION.
        useInputs : TYPE, optional
            DESCRIPTION. The default is True.
        withoutStorage : TYPE, optional
            DESCRIPTION. The default is True.
        minSum : TYPE, optional
            DESCRIPTION. The default is .1.
        othersMax_rel : TYPE, optional
            DESCRIPTION. The default is 0.05.
        plotAsPlotly : TYPE, optional
            DESCRIPTION. The default is False.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        unit : TYPE, optional
            DESCRIPTION. The default is 'FlowHours'.
  
        Returns
        -------
        nice plot ;)
        '''

        # if not a list of str yet, transform to list:
        if isinstance(busesOrComponents, str):
            busesOrComponents =  [busesOrComponents]

        in_flows_all = []            
        out_flows_all = []
        for busOrComponent in busesOrComponents:          
            (in_flows, out_flows) = self.getFlowsOf(busOrComponent)
            in_flows_all += in_flows
            out_flows_all += out_flows
            
        if useInputs:
            flows = in_flows_all
        else:
            flows = out_flows_all
        
        # delete flows which belong to storage (cause they disturbing the plot):
        if withoutStorage:
            # Umsetzung not nice, aber geht!
            allowed_i = [] 
            for i in range(len(flows)):        
                aFlow = flows[i]        
                if not aFlow.belongToStorage():
                    allowed_i.append(i)
            flows = [flows[i] for i in allowed_i] # Gekürzte liste
          
        
        sums = np.array([])
        labels = []
        colors = []
        totalSum = 0
        for aFlow in flows:
          totalSum +=sum(aFlow.results['val'])
          
        others_Sum = 0
        for aFlow in flows:
          aSum = sum(aFlow.results['val'])
          if aSum >minSum:
            if aSum/totalSum < othersMax_rel:
              others_Sum += aSum
            else:
              sums=np.append(sums,aSum)
              labels.append(aFlow.comp + '.' + aFlow.label)
              colors.append(aFlow.color)
              
        
        if others_Sum >0:
          sums = np.append(sums,others_Sum)
          labels.append('others')
          colors.append('#AAAAAA')# just a grey
        
        aText = "total: {:.0f}".format(sum(sums)) + ' ' + unit 
        
        if title is None:
            title=",".join(busesOrComponents)
            if useInputs:
              title+= ' (supply)'
            else:
              title+= ' (usage)'
            
        
        
        
        def plot_matplotlib(sums, labels, title, aText):
            fig = plt.figure()
            ax = fig.add_subplot()
            plt.title(title)
            plt.pie(sums/sum(sums), labels = labels)            
            fig.text(0.95, 0.05, aText,
                  verticalalignment='top', horizontalalignment='center',
                  transform=ax.transAxes,
                  color='black', fontsize=10)
            # ax.text(0.95, 0.98, aText,
            #       verticalalignment='top', horizontalalignment='right',
            #       transform=ax.transAxes,
            #       color='black', fontsize=10)
            plt.show()
        
        def plot_plotly(sums, labels,title, aText, colors):            
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(labels=labels, values=sums, marker_colors = colors)])
            fig.update_layout(title_text = title,
                              annotations = [dict(text=aText, x=0.95, y=0.05, font_size=20, align = 'right', showarrow=False)],
                              )
            fig.show()
            
        if plotAsPlotly:
          plot_plotly    (sums, labels, title, aText, colors)
        else:
          plot_matplotlib(sums, labels, title, aText)
                           
            
    def plotInAndOuts(self, busOrComponent, stacked = False, renderer='browser', minFlowHours=0.1, plotAsPlotly = False, title = None, outFlowCompsAboveXAxis=None, sortBy = None):
        '''      
        Parameters
        ----------
        busOrComponent : TYPE
            DESCRIPTION.
        stacked : TYPE, optional
            DESCRIPTION. The default is False.
        renderer : 'browser', 'svg',...
        
        minFlowHours : TYPE, optional
            min absolute sum of Flows for Showing curve. The default is 0.1.
        plotAsPlotly : boolean, optional
        
        title : str, optional
            if None, then automatical title is used    
            
        outFlowCompsAboveXAxis : components
            End-Components of outflows, which should be shown separately above x-Axis, i.g. heat-load
            
        sortBy : component or None, optional    
            Component-Flow which should be used for sorting the timeseries ("Jahresdauerlinie")
        '''
        
        if not (busOrComponent in self.results.keys()):
            raise Exception(str(busOrComp) + 'is no valid bus or component name')
        
        import plotly.io as pio            
        pio.renderers.default = renderer # 'browser', 'svg',...
    
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        (in_flows, out_flows) = self.getFlowsOf(busOrComponent)           
               
        # sorting:
        if sortBy is not None:
            # find right flow:
            (ins, outs) = self.getFlowsOf(busOrComponent,sortBy)
            flowForSort = (ins+outs)[0] # dirty!
            # find index sequence
            indexSeq = np.argsort(flowForSort.results['val']) # ascending
            indexSeq = indexSeq[::-1] # descending
        else:
            indexSeq = None
            
        # extract outflows above x-Axis:
        out_flows_above_x_axis = []
        if outFlowCompsAboveXAxis is not None:
            for flow in out_flows:
                if flow.to_node in outFlowCompsAboveXAxis:
                    out_flows.remove(flow)
                    out_flows_above_x_axis.append(flow)
        
        # Inputs:
        y_in, y_in_colors = self.__get_Values_As_DataFrame(in_flows, self.timeSeriesWithEnd, self.dtInHours, minFlowHours, indexSeq=indexSeq)
        # Outputs; als negative Werte interpretiert:
        y_out, y_out_colors = self.__get_Values_As_DataFrame(out_flows,self.timeSeriesWithEnd, self.dtInHours, minFlowHours, indexSeq=indexSeq)
        y_out = -1 * y_out 

        y_out_aboveX, y_above_colors = self.__get_Values_As_DataFrame(out_flows_above_x_axis,self.timeSeriesWithEnd, self.dtInHours, minFlowHours, indexSeq=indexSeq)

        # if hasattr(self, 'excessIn')  and (self.excessIn is not None):
        if 'excessIn' in self.results[busOrComponent].keys():
            # in and out zusammenfassen:
            
            excessIn = self.results[busOrComponent]['excessIn'] 
            excessOut = - self.results[busOrComponent]['excessOut']
            
            if flix_results.isGreaterMinFlowHours(excessIn, self.dtInHours, minFlowHours):        
                y_in['excess_in']   = excessIn
                y_in_colors.append('#FF0000')
            if flix_results.isGreaterMinFlowHours(excessOut, self.dtInHours, minFlowHours):        
                y_out['excess_out'] = excessOut
                y_out_colors.append('#FF0000')
                
          

    
    
        # wenn title nicht gegeben
        if title is None:
            title = busOrComponent + ': '+ ' in (+) and outs (-)' + ' [' + self.label + ']'
        yaxes_title = 'Flow'
        yaxes2_title = 'charge state'
    
    
        def plotY_plotly(y_pos, y_neg, y_pos_separat, title, yaxes_title, yaxes2_title, y_pos_colors, y_neg_colors, y_above_colors):
    
            ## Flows:
            # fig = go.Figure()
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # def plotlyPlot(y_in, y_out, stacked, )
            # input:
            y_pos_colors = iter(y_pos_colors)
            y_neg_colors = iter(y_neg_colors)
            y_above_colors = iter(y_above_colors)
            for column in y_pos.columns:
                aColor = next(y_pos_colors)
                # if isGreaterMinAbsSum(y_in[column]):
                if stacked :
                    fig.add_trace(go.Scatter(x=y_pos.index, y=y_pos[column],stackgroup='one',line_shape ='hv',name=column, line_color=aColor))
                else:
                    fig.add_trace(go.Scatter(x=y_pos.index, y=y_pos[column],line_shape ='hv',name=column, line_color = aColor))
            # output:
            for column in y_neg.columns:
                aColor = next(y_neg_colors)
                # if isGreaterMinAbsSum(y_out[column]):
                if stacked :
                    fig.add_trace(go.Scatter(x=y_neg.index, y=y_neg[column],stackgroup='two',line_shape ='hv',name=column, line_color = aColor))
                else:
                    fig.add_trace(go.Scatter(x=y_neg.index, y=y_neg[column],line_shape ='hv',name=column, line_color = aColor))
            
            # output above x-axis:
            for column in y_pos_separat:
                aColor = next(y_above_colors)
                fig.add_trace(go.Scatter(x=y_pos_separat.index, y=y_pos_separat[column],line_shape ='hv',line=dict(dash='dash', width = 4) , name=column, line_color = aColor))
            
            
            # ## Speicherverlauf auf Sekundärachse:
            # # Speicher finden:
            # setOfStorages = set()
            # # for aFlow in self.inputs + self.outputs:
            # for acomp in self.modBox.es.allMEsOfFirstLayerWithoutFlows:
            #   if acomp.__class__.__name__ == 'cStorage': # nicht schön, da cStorage hier noch nicht bekannt
            #     setOfStorages.add(acomp)      
            # for aStorage in setOfStorages:
            #   chargeState = aStorage.mod.var_charge_state.getResult()
            #   fig.add_trace(go.Scatter(x=timeSeriesWithEnd, y=chargeState, name=aStorage.label+'.chargeState',line_shape='linear',line={'dash' : 'dash'} ),secondary_y = True)
              
        
            # fig.update_layout(title_text = title,
            #                   xaxis_title = 'Zeit',
            #                   yaxis_title = 'Leistung [kW]')
            #                   # yaxis2_title = 'charge state')      
            fig.update_xaxes(title_text="Zeit")
            fig.update_yaxes(title_text=yaxes_title, secondary_y=False)
            fig.update_yaxes(title_text=yaxes2_title, secondary_y=True)      
            fig.update_layout(title=title)
            fig.show()
        
        def plotY_matplotlib(y_pos, y_neg, y_pos_separat, title, yaxes_title, yaxes2_title):
            # Verschmelzen:
            y = pd.concat([y_pos,y_neg],axis=1)
                  
            fig, ax = plt.subplots(figsize = (18,10)) 
            
            # separate above x_axis
            if len(y_pos_separat.columns) > 0 :                    
                for column in y_pos_separat.columns:
                    ax.plot(y_pos_separat.index, y_pos_separat[column], '--', drawstyle='steps-post', linewidth=3, label = column)

            # gestapelt:
            if stacked :               
                helpers.plotStackedSteps(ax, y) # legende here automatically                
            # normal:
            else:
                y.plot(drawstyle='steps-post', ax=ax)
                

            plt.legend(fontsize=22, loc="upper center",  bbox_to_anchor=(0.5, -0.2), markerscale=2,  ncol=3, frameon=True, fancybox= True, shadow=True)
            # plt.legend(loc="upper center",  bbox_to_anchor=(0.5, -0.2), fontsize=22, ncol=3, frameon=True, shadow= True, fancybox=True) 

                              
            fig.autofmt_xdate()
           
            xfmt = mdates.DateFormatter('%d-%m')
            ax.xaxis.set_major_formatter(xfmt)
            
            plt.title(title, fontsize= 30)
            plt.xlabel('Zeit - Woche [h]', fontsize = 'xx-large')                                                 ### x-Achsen-Titel                     
            plt.ylabel(yaxes_title ,fontsize = 'xx-large')                                            ### y-Achsen-Titel  = Leistung immer
            plt.grid()
            plt.show()      
            
            
          
        if plotAsPlotly:
            plotY_plotly(y_in, y_out, y_out_aboveX, title, yaxes_title, yaxes2_title,
                         y_in_colors, y_out_colors, y_above_colors)
        else:
            plotY_matplotlib(y_in, y_out, y_out_aboveX, title, yaxes_title, yaxes2_title)
                             
              

# self.results[]
## ideen: kosten-übersicht, JDL, Torten-Diag für Busse