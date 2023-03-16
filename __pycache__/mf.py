# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:51:58 2023

@author: nishi
"""


'''************************************************************************************************************************************'''
def fn_email_positions(strat_names):
    df_temnp_delta                                              = Database['AGGREGATE']['Positions DELTA'].sort_values(['TYPE', 'CLASS', 'COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1)
    # df_temnp_delta                                              = integer_separator(df_temnp_delta, strat_names)
    
    df_temp_pos                                                 = Database['AGGREGATE']['Positions'].sort_values(['TYPE', 'CLASS', 'COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1)
    # df_temp_pos                                                 = integer_separator(df_temp_pos, strat_names)
    
    df_temp_prev_pos                                            = Database['AGGREGATE']['Positions Previous'].sort_values(['TYPE', 'CLASS', 'COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1).drop(['Current Risk'], axis = 0)
    # df_temp_prev_pos                                            = integer_separator(df_temp_prev_pos, strat_names)
    
    
    df_dict_now,names_now                                       = segregator(df_temnp_delta.copy(), 'CLASS', 'COMPOSITE BPS', 0)
    names_now                                                   = ['DELTA ' + s  for s in names_now]
    
    Emailer.send_email(
        dfs=df_dict_now,
        excel_file_names=names_now,
        subject='POSITIONS DELTA',
        send_to_list=email_list
    )

    df_dict_now,names_now                                       = segregator(df_temp_pos.copy(), 'CLASS', 'COMPOSITE BPS', 0)
    names_now                                                   = ['CURRENT POS ' + s  for s in names_now]
    
    Emailer.send_email(
        dfs=df_dict_now,
        excel_file_names=names_now,
        subject='POSITIONS CURRENT',
        send_to_list=email_list
    )               
    
    df_dict_now,names_now                                       = segregator(df_temp_prev_pos.copy(), 'CLASS', 'COMPOSITE BPS', 0)
    names_now                                                   = ['PREVIOUS POS ' + s  for s in names_now]
    
    Emailer.send_email(
        dfs=df_dict_now,
        excel_file_names=names_now,
        subject='POSITIONS PREVIOUS',
        send_to_list=email_list
    )           

def fn_email_bps(strat_names):
    df_temnp_delta                                              = Database['AGGREGATE']['BPS'].sort_values(['TYPE', 'CLASS', 'COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1)
    df_temnp_delta.drop('Current Risk', inplace = True)

    df_temnp_delta['Z Score']                                   = Database['COMPOSITE']['Boll_Bands']['Z-Score: 252'].iloc[-1]
    # df_temnp_delta                                              = df_temnp_delta[[ 'CLASS', 'COMPOSITE BPS' ,'Z Score'] + strat_names]

    df_dict_now,names_now                                       = segregator(df_temnp_delta.copy(), 'CLASS', 'COMPOSITE BPS', 0)
    names_now                                                   = ['BPS ' + s  for s in names_now]
    try:
        old_emailer.EmailRangeHandler(dfs_to_email = df_dict_now, dfs_header   = names_now, outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='CURRENT BPS',mail_list=email_list,mail_cc_list=[])      
        # email_dataframes(df_dict_now , names_now ,'CURRENT BPS' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       
        
    except:
        print('Error using emailer')
    
    #util.kill_xl()

'''************************************************************************************************************************************'''
def fn_email_risk(strat_names):    
    ######################################### EMAILING STANDALONE VOL ###############################

    # df_stand_vol                                                    = Database['AGGREGATE']['StandAlone_Vol'].sort_values(['TYPE', 'CLASS', 'COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1) 
    # df_stand_vol_sign                                               = Database['AGGREGATE']['Signed StandAlone_Vol'].sort_values(['TYPE', 'CLASS', 'COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1)
    # df_vol_agg                                                      = Database['AGGREGATE']['Class Signed StandAlone Vol']
    fields                                                          = strat_names
    
    # plot_pie_chart(df_stand_vol.groupby(['CLASS']).sum(),df_stand_vol.groupby(['CLASS']).sum().index,'COMPOSITE BPS', 'STANDALONE VOL' , 8)

    # df_dict_now = []
    # df_dict_now.append(df_stand_vol_sign.sort_values('COMPOSITE BPS', ascending = False).head(5))
    # df_dict_now.append(df_stand_vol_sign.sort_values('COMPOSITE BPS', ascending = False).tail(5))
    # df_dict_now.append(df_vol_agg)
    # df_dict_now.append(df_stand_vol_sign.sort_values('COMPOSITE BPS', ascending = False))
    
    # names_now=['Signed Top 5','Signed Bottom 5' , 'Class Signed StandAlone Vol', ' Signed StandAlone_Vol' ]
    # old_emailer.EmailRangeHandler(dfs_to_email = df_dict_now, dfs_header   = names_now, outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Standalone Vol',mail_list=email_list,mail_cc_list=[])           
    

    ######################################### EMAILING RISK CONTRIBUTIONS ###############################
    df_temp_risk                                                    = Database['AGGREGATE']['Risk Contribution'].sort_values(['COMPOSITE BPS'] , ascending = False).drop(['TYPE','RISK_TICKER', 'NO_ADJ', 'MAP'], axis = 1)
    df_temp_risk[fields]                                            = df_temp_risk[fields]*100
    df_temp_risk_head                                               = df_temp_risk.head(5)
    df_temp_risk_tail                                               = df_temp_risk.tail(5)


    df_dict_now = []
    df_dict_now.append(df_temp_risk_head)
    df_dict_now.append(df_temp_risk_tail)
    df_dict_now.append(df_temp_risk)
    
    names_now=['LONG 5' , 'SHORT 5',' ALL Risk Contribution' ]
    try:
        old_emailer.EmailRangeHandler(dfs_to_email = df_dict_now, dfs_header   = names_now, outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Risk Contribution Scorecards',mail_list=email_list,mail_cc_list=[])    
        # email_dataframes(df_dict_now , names_now ,'Risk Contribution Scorecards' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       
        
    except:
        print('Error using emailer')

'''************************************************************************************************************************************'''
def fn_email_strat_risk(df_rc_strat , df_pc_strat, df_mc_strat , df , current_risk , df_rc_unit_strat , df_pc_unit_strat, df_mc_unit_strat , df_rc_opt , df_pc_opt, df_mc_opt, optimized_risk):    
    # df = Database['Returns'].copy()
    ######################################### EMAILING RISK CONTRIBUTIONS ###############################

    temp                                                                = df['ALL_IN'].copy()
    temp['Composite']                                                   = df['ALL_IN'].sum(axis = 1)
    yearly_strat_std_as_is                                              = temp.iloc[-252:].std()*np.sqrt(252)
    yearly_strat_std_as_is                                              = pd.DataFrame(yearly_strat_std_as_is)*100
    yearly_strat_std_as_is.columns                                      = ['Vol Realized']
    yearly_strat_std_as_is                                              = yearly_strat_std_as_is.transpose()
         
    temp                                                                = df['ALL IN Unit Returns'].copy()
    temp['Composite']                                                   = df['ALL IN Unit Returns'].sum(axis = 1)
    yearly_strat_std_unit                                               = temp.iloc[-252:].std()*np.sqrt(252)
    yearly_strat_std_unit                                               = pd.DataFrame(yearly_strat_std_unit)*100
    yearly_strat_std_unit.columns                                       = ['Vol Unit Risk']
    yearly_strat_std_unit                                               = yearly_strat_std_unit.transpose()
    
    
    temp                                                                = df['ALL IN Current Risk Returns'].copy()
    temp['Composite']                                                   = df['ALL IN Current Risk Returns'].sum(axis = 1)
    yearly_strat_std_cur_risk                                           = temp.iloc[-252:].std()*np.sqrt(252)
    yearly_strat_std_cur_risk                                           = pd.DataFrame(yearly_strat_std_cur_risk)*100
    yearly_strat_std_cur_risk.columns                                   = ['Vol Current Risk']
    yearly_strat_std_cur_risk                                           = yearly_strat_std_cur_risk.transpose()
    
    temp                                                                = current_risk.copy()
    cur_risk                                                            = pd.DataFrame.from_dict(temp , orient = 'index')
    cur_risk.columns                                                    = ['Current Deployed Risk']
    cur_risk                                                            = cur_risk.transpose()
    cur_risk                                                            = cur_risk[df_mc_strat.index]
    
    temp                                                                = optimized_risk.copy()
    opt_risk                                                            = pd.DataFrame.from_dict(temp , orient = 'index')
    opt_risk.columns                                                    = ['Optimized Risk']
    opt_risk                                                            = opt_risk.transpose()
    opt_risk                                                            = opt_risk[df_mc_opt.index]


    df_rc                                                               = pd.DataFrame(df_rc_strat)*100
    df_rc.loc['Total Vol']                                              = df_rc['Risk Contri'].sum()
    df_rc                                                               = df_rc.transpose()
    df_pc                                                               = pd.DataFrame(df_pc_strat)*100
    df_pc.loc['Total']                                                  = df_pc['Per Contri'].sum()
    df_pc                                                               = df_pc.transpose()    
    df_bc                                                               = pd.DataFrame(df_mc_strat)*100
    df_bc                                                               = df_bc.transpose()    

    df_rc_unit                                                               = pd.DataFrame(df_rc_unit_strat)*100
    df_rc_unit.loc['Total Vol']                                              = df_rc_unit['Risk Contri'].sum()
    df_rc_unit                                                               = df_rc_unit.transpose()
    df_pc_unit                                                               = pd.DataFrame(df_pc_unit_strat)*100
    df_pc_unit.loc['Total']                                                  = df_pc_unit['Per Contri'].sum()
    df_pc_unit                                                               = df_pc_unit.transpose()    
    df_bc_unit                                                               = pd.DataFrame(df_mc_unit_strat)*100
    df_bc_unit                                                               = df_bc_unit.transpose()    
    
    df_rc_opt                                                               = pd.DataFrame(df_rc_opt)*100
    df_rc_opt.loc['Total Vol']                                              = df_rc_opt['Risk Contri'].sum()
    df_rc_opt                                                               = df_rc_opt.transpose()
    df_pc_opt                                                               = pd.DataFrame(df_pc_opt)*100
    df_pc_opt.loc['Total']                                                  = df_pc_opt['Per Contri'].sum()
    df_pc_opt                                                               = df_pc_opt.transpose()    
    df_mc_opt                                                               = pd.DataFrame(df_mc_opt)*100
    df_mc_opt                                                               = df_mc_opt.transpose()    

    df_dict_now = []
    df_dict_now.append(df_rc)
    df_dict_now.append(df_pc)
    df_dict_now.append(df_bc)

    df_dict_now.append(df_rc_unit)
    df_dict_now.append(df_pc_unit)
    df_dict_now.append(df_bc_unit)

    df_dict_now.append(df_rc_opt)
    df_dict_now.append(df_pc_opt)
    df_dict_now.append(df_mc_opt)

    df_dict_now.append(yearly_strat_std_as_is)
    df_dict_now.append(yearly_strat_std_unit)
    df_dict_now.append(yearly_strat_std_cur_risk)
    df_dict_now.append(cur_risk)
    df_dict_now.append(opt_risk)
    
    names_now=['Risk Contribution' , 'Percentage Risk Contribution','Marginal Risk Contribution' , 'Unit Risk Contribution' , 'Unit Percentage Risk Contribution','Unit Marginal Risk Contribution' 
               , 'Optimized Risk Contribution' , 'Optimized Percentage Risk Contribution','Optimized Marginal Risk Contribution' 
               , 'Vol Realized' , 'Vol Unit Risk', 'Vol Current Risk' , 'Current Deployed Risk', 'Optimized Risk']
    try:
        # old_emailer.EmailRangeHandler(dfs_to_email = df_dict_now, dfs_header   = names_now, outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Risk Contribution Strategy',mail_list=email_list,mail_cc_list=[])     
        email_dataframes(df_dict_now , names_now ,'Risk Contribution Strategy' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       
        
    except:
        print('Error using emailer')    

'''************************************************************************************************************************************'''



def fn_email_performance(Database , strat_names_2):    
    df_1D                                                           = Database['AGGREGATE']['PnL ScoreCard']['DAY_CHANGE'].sort_values(['TYPE', 'CLASS', 'COMPOSITE'] , ascending = False).sort_values('COMPOSITE', ascending = False) 
    df_1D['Comp BPS']                                               = Database['AGGREGATE']['BPS']['COMPOSITE BPS']
    df_1D['Trend BPS']                                              = Database['AGGREGATE']['BPS']['TREND']
    df_1D['Carry BPS']                                              = Database['AGGREGATE']['BPS']['CARRY']
    df_1D['ECS BPS']                                                = Database['AGGREGATE']['BPS']['ECS']
    df_1D['FICS BPS']                                               = Database['AGGREGATE']['BPS']['FICS']
    df_1D['EMFX BPS']                                               = Database['AGGREGATE']['BPS']['EMFX']
    df_1D['G10FX BPS']                                              = Database['AGGREGATE']['BPS']['G10FX']
    df_1D['TRH BPS']                                                = Database['AGGREGATE']['BPS']['TRH']
    df_1D['EDF BPS']                                                = Database['AGGREGATE']['BPS']['EDF']
    df_1D['IDMO BPS']                                               = Database['AGGREGATE']['BPS']['IDMO']
    df_1D = df_1D[['TYPE', 'CLASS', 'ACTIVE_TICKER','Comp BPS',  'COMPOSITE','Trend BPS',  'TREND', 'Carry BPS' , 'CARRY', 'ECS BPS','ECS','FICS BPS', 'FICS', 'EMFX BPS', 'EMFX', 'G10FX BPS', 'G10FX','TRH BPS', 'TRH', 'EDF BPS','EDF' , 'IDMO BPS','IDMO']]
    df_MTD                                                          = Database['AGGREGATE']['PnL ScoreCard']['MTD'].sort_values(['TYPE', 'CLASS', 'COMPOSITE'] , ascending = False).sort_values('COMPOSITE', ascending = False)  
    df_MTD['Comp BPS']                                               = Database['AGGREGATE']['BPS']['COMPOSITE BPS']
    df_MTD['Trend BPS']                                              = Database['AGGREGATE']['BPS']['TREND']
    df_MTD['Carry BPS']                                              = Database['AGGREGATE']['BPS']['CARRY']
    df_MTD['ECS BPS']                                                = Database['AGGREGATE']['BPS']['ECS']
    df_MTD['FICS BPS']                                               = Database['AGGREGATE']['BPS']['FICS']
    df_MTD['EMFX BPS']                                               = Database['AGGREGATE']['BPS']['EMFX']
    df_MTD['G10FX BPS']                                              = Database['AGGREGATE']['BPS']['G10FX']
    df_MTD['TRH BPS']                                                = Database['AGGREGATE']['BPS']['TRH']
    df_MTD['EDF BPS']                                                = Database['AGGREGATE']['BPS']['EDF']
    df_MTD['IDMO BPS']                                               = Database['AGGREGATE']['BPS']['IDMO']
    df_MTD = df_MTD[['TYPE', 'CLASS', 'ACTIVE_TICKER','Comp BPS',  'COMPOSITE','Trend BPS',  'TREND', 'Carry BPS' , 'CARRY', 'ECS BPS','ECS','FICS BPS', 'FICS', 'EMFX BPS', 'EMFX', 'G10FX BPS', 'G10FX','TRH BPS', 'TRH', 'EDF BPS','EDF', 'IDMO BPS','IDMO']]

    df_YTD                                                          = Database['AGGREGATE']['PnL ScoreCard']['YTD'].sort_values(['TYPE', 'CLASS', 'COMPOSITE'] , ascending = False).sort_values('COMPOSITE', ascending = False)  
    
    fields                                                          = ['COMPOSITE', 'TREND', 'CARRY', 'ECS','FICS', 'EMFX', 'G10FX', 'TRH', 'EDF' , 'IDMO']
    temp_df_1D                                                      = integer_separator(df_1D, fields)
    temp_df_MTD                                                     = integer_separator(df_MTD, fields)
    temp_df_YTD                                                     = integer_separator(df_YTD, fields)

    df_1D_class                                                     = df_1D.groupby(['CLASS']).sum()
    df_1D_class                                                     = integer_separator(df_1D_class, strat_names_2)
    df_MTD_class                                                    = df_MTD.groupby(['CLASS']).sum()
    df_MTD_class                                                    = integer_separator(df_MTD_class, strat_names_2)
    df_YTD_class                                                    = df_YTD.groupby(['CLASS']).sum()
    df_YTD_class                                                    = integer_separator(df_YTD_class, strat_names_2)
    
    df1D_total                                                      = pd.DataFrame(temp_df_1D.loc['TOTAL',:]).transpose().drop(['TYPE', 'CLASS', 'ACTIVE_TICKER'], axis = 1)
    dfMTD_total                                                     = pd.DataFrame(temp_df_MTD.loc['TOTAL',:]).transpose().drop(['TYPE', 'CLASS', 'ACTIVE_TICKER'], axis = 1)
    dfYTD_total                                                     = pd.DataFrame(temp_df_YTD.loc['TOTAL',:]).transpose().drop(['TYPE', 'CLASS', 'ACTIVE_TICKER'], axis = 1)

    plot_bar_chart(df_1D,'TOTAL',fields, 'Strategies' , 'Daily P&L')
    plot_bar_chart(df_MTD,'TOTAL',fields, 'Strategies' , 'Monthly P&L')
    plot_bar_chart(df_YTD,'TOTAL',fields, 'Strategies' , 'Yearly P&L')
        
    if send_pnl_email:
        dfMTD_total.to_csv('MTD_total.csv')
        temp_df             = temp_df_MTD[['COMPOSITE']]
        temp_df             = temp_df[temp_df['COMPOSITE'].str.replace(',','').astype(float).abs() > 1000000]
        temp_df.sort_values(by='COMPOSITE', inplace=True)
        temp_df             = temp_df[temp_df.index != 'TOTAL']
        temp_df.to_csv('MTD_all.csv')

        try:
            old_emailer.EmailRangeHandler(dfs_to_email = [df1D_total , temp_df_1D.drop('TOTAL', axis = 0).head(5) ,temp_df_1D.drop('TOTAL', axis = 0).tail(5) , df_1D_class , temp_df_1D] ,
                                  dfs_header   = ['Daily Total', 'Top Daily Gainers','Top Daily Losers','ALL Daily GROUPED' , 'ALL Daily P&L'], outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Daily P&L',mail_list=email_list,mail_cc_list=[])           
            # email_dataframes( [df1D_total , temp_df_1D.drop('TOTAL', axis = 0).head(5) ,temp_df_1D.drop('TOTAL', axis = 0).tail(5) , df_1D_class , temp_df_1D] 
            #                  ,  ['Daily Total', 'Top Daily Gainers','Top Daily Losers','ALL Daily GROUPED' , 'ALL Daily P&L']
            #                  ,'Daily P&L' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       

        except:
            print('Error using emailer')
        try:
            old_emailer.EmailRangeHandler(dfs_to_email = [dfMTD_total, temp_df_MTD.drop('TOTAL', axis = 0).head(5) ,temp_df_MTD.drop('TOTAL', axis = 0).tail(5) , df_MTD_class, temp_df_MTD] ,
                                  dfs_header   = ['Monthly Total', 'Top Monthly Gainers','Top Monthly Losers', 'ALL MTD GROUPED' ,'ALL Monthly P&L'], outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Monthly P&L',mail_list=email_list,mail_cc_list=[])           
            # email_dataframes(  [dfMTD_total, temp_df_MTD.drop('TOTAL', axis = 0).head(5) ,temp_df_MTD.drop('TOTAL', axis = 0).tail(5) , df_MTD_class, temp_df_MTD]
            #                  ,   ['Monthly Total', 'Top Monthly Gainers','Top Monthly Losers', 'ALL MTD GROUPED' ,'ALL Monthly P&L']
            #                  ,'Monthly P&L' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       
        except:
            print('Error using emailer')
        try:
            old_emailer.EmailRangeHandler(dfs_to_email = [dfYTD_total,temp_df_YTD.drop('TOTAL', axis = 0).head(5) ,temp_df_YTD.drop('TOTAL', axis = 0).tail(5) , df_YTD_class, temp_df_YTD] ,
                                  dfs_header   = ['Yearly Total','Top Yearly Gainers','Top Yearly Losers', 'ALL YTD GROUPED' ,'ALL Yearly P&L'], outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='YTD P&L',mail_list=email_list,mail_cc_list=[])
            # email_dataframes( [dfYTD_total,temp_df_YTD.drop('TOTAL', axis = 0).head(5) ,temp_df_YTD.drop('TOTAL', axis = 0).tail(5) , df_YTD_class, temp_df_YTD]
            #                  ,   ['Yearly Total','Top Yearly Gainers','Top Yearly Losers', 'ALL YTD GROUPED' ,'ALL Yearly P&L']
            #                  ,'YTD P&L' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       
        except:
            print('Error using emailer')
            

        
        
    return df_1D , df_MTD , df_YTD


def fn_email_strat_perf(Database , df_LCL_Pnl_by_USD_daily , strat_names_2 , NAV_Start_of_Year):
    df_dict_now             = []
    names_now               = []           
    df                      = pd.DataFrame( index =  Database['PnL']['COMPOSITE']['Total PnL'].index , columns =  ['COMPOSITE', 'TREND', 'CARRY', 'ECS','FICS', 'EMFX', 'G10FX', 'TRH', 'EDF', 'IDMO'])

    df_LCL_Pnl_by_USD_daily['COMPOSITE'] = df_LCL_Pnl_by_USD_daily.sum(axis = 1)
    df_LCL_FX               = Database['PnL']['COMPOSITE']['LCL PnL'].copy()
    df_returns              = Database['Returns']['ALL_IN'].copy()
    df_returns['COMPOSITE'] = Database['Returns']['ALL_IN'].sum(axis = 1)
        
    for strategy_name in strat_names_2:
        df[strategy_name]         = Database['PnL'][strategy_name]['Total PnL']

    df.fillna(0, inplace = True)
    
    temp_all_fact                   = df.copy()
    df_temnp_delta_fact             = pd.DataFrame(data = np.nan, index = temp_all_fact.columns, columns = ['1D', '1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '5Y'])
    df_temnp_delta_fact['1D']       = temp_all_fact.iloc[-1] 
    df_temnp_delta_fact['1W']       = temp_all_fact.iloc[-5:].sum() 
    df_temnp_delta_fact['1M']       = temp_all_fact.iloc[-22:].sum()  
    df_temnp_delta_fact['2M']       = temp_all_fact.iloc[-44:].sum()  
    df_temnp_delta_fact['3M']       = temp_all_fact.iloc[-66:].sum()  
    df_temnp_delta_fact['6M']       = temp_all_fact.iloc[-132:].sum()  
    df_temnp_delta_fact['9M']       = temp_all_fact.iloc[-198:].sum()  
    df_temnp_delta_fact['1Y']       = temp_all_fact.iloc[-252:].sum()  
    df_temnp_delta_fact['2Y']       = temp_all_fact.iloc[-504:].sum()  
    df_temnp_delta_fact['3Y']       = temp_all_fact.iloc[-756:].sum()  
    df_temnp_delta_fact['5Y']       = temp_all_fact.iloc[-1260:].sum()  
                           
    df_temnp_delta_fact             = df_temnp_delta_fact.sort_values(['1D', '1W', '1M', '2M', '3M'] , ascending = False)
    # df_temnp_delta_fact             = integer_separator(df_temnp_delta_fact, df_temnp_delta_fact.columns)
    df_temnp_delta_fact             = df_temnp_delta_fact.drop('COMPOSITE').append(df_temnp_delta_fact.loc['COMPOSITE'])        
        
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('Strategy Performance')          

    df_2                            = temp_all_fact.resample('BM').sum().iloc[-12:]
    # df_2                            = integer_separator(df_2, df_2.columns)

    df_dict_now.append(df_2) 
    names_now.append('Performance Monthly')  
    
    df_FX                           = df_LCL_FX.resample('BM').sum().iloc[-12:]
    # df_FX                           = integer_separator(df_FX, df_FX.columns)

    df_dict_now.append(df_FX) 
    names_now.append('Local Currency Realized Monthly P&L')  
    
    df_FX_USD                       = df_LCL_Pnl_by_USD_daily.resample('BM').sum().iloc[-12:]
    # df_FX_USD                       = integer_separator(df_FX_USD, df_FX_USD.columns)

    df_dict_now.append(df_FX_USD) 
    names_now.append('Local Currency Realized Monthly P&L -> USD equivalent')      

    
    df_6                            = temp_all_fact.resample('Y').sum().iloc[-12:]
    # df_6                            = integer_separator(df_6, df_6.columns)

    df_dict_now.append(df_6) 
    names_now.append('Performance Yearly')    

    df_FX_Y                         = df_LCL_FX.resample('Y').sum().iloc[-12:]
    # df_FX_Y                         = integer_separator(df_FX_Y, df_FX_Y.columns)

    df_dict_now.append(df_FX_Y) 
    names_now.append('Local Currency Realized Yearly')  

    df_FX_USD_Y                      = df_LCL_Pnl_by_USD_daily.resample('Y').sum().iloc[-12:]
    # df_FX_USD_Y                      = integer_separator(df_FX_USD_Y, df_FX_USD_Y.columns)

    df_dict_now.append(df_FX_USD_Y) 
    names_now.append('Local Currency Realized Yearly P&L -> USD equivalent') 
    
    current_year                    = str(date.today().year)
    df_7                            = temp_all_fact.loc[current_year]
    df_7['CUMULATIVE']              = df_7['COMPOSITE'].cumsum()
    hwm                             = (df_7['CUMULATIVE'] + NAV_Start_of_Year).cummax()
    nav                             = df_7['CUMULATIVE'] + NAV_Start_of_Year
    mdd_percentage                  = nav/hwm - 1
    
    df_7_cum                        = (1+(temp_all_fact.loc[current_year]/NAV_Start_of_Year)).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100
    
    df_7                            = df_7.sort_index(ascending = False)
    df_vol                          = temp_all_fact.rolling(window = 60).std()*np.sqrt(252).round()
    df_7['60D VOL']                 = df_vol['COMPOSITE']

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD' , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns = strat_names_2 )
    df_9.loc['Returns']             = df_8.sum()
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.sum().round())/((df_8.std()*np.sqrt(252)).round())

    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])
        
    # df_7                            = integer_separator(df_7, df_7.columns)
    df_7['MAX DD']                  = round(mdd_percentage*100,2)
    df_7                            = df_7[ ['CUMULATIVE' ,'MAX DD' ,  '60D VOL'] + strat_names_2]



    df_dict_now.append(df_9) 
    names_now.append('Stats')   

    annual_return                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_6.index.year)], columns = df_returns.columns)
    annual_vol                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_6.index.year)], columns = df_returns.columns)

    annual_sharpe                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_6.index.year)], columns = df_returns.columns)
    annual_mdd                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_6.index.year)], columns = df_returns.columns)

    for year in annual_sharpe.index:
        annual_return.loc[year]     = df_returns.loc[year].sum()*100
        annual_vol.loc[year]        = df_returns.loc[year].std()*np.sqrt(252)*100
        annual_sharpe.loc[year]     = (df_returns.loc[year].sum())/(df_returns.loc[year].std()*np.sqrt(252))
        df_cum                      = (1+df_returns.loc[year]).cumprod()
        df_HWM                      = df_cum.cummax()
        df_DD                       = df_cum/df_HWM - 1
        annual_mdd.loc[year]        = df_DD.min()*100

    # annual_sharpe                    = integer_separator(annual_sharpe, annual_sharpe.columns)

    df_dict_now.append(annual_return) 
    names_now.append('Annual Realized Return')  
    
    df_dict_now.append(annual_vol) 
    names_now.append('Annual Realized Vol')  

    df_dict_now.append(annual_sharpe) 
    names_now.append('Annual Realized Sharpe')  

    # annual_mdd                      = integer_separator(annual_mdd, annual_mdd.columns)

    df_dict_now.append(annual_mdd) 
    names_now.append('Annual Max Drawdown')  

    
    df_dict_now.append(df_7) 
    names_now.append('Daily Performance History') 
    
    try:
        # old_emailer.EmailRangeHandler(dfs_to_email = df_dict_now, dfs_header = names_now, outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Strategy Performance',mail_list=['nishikant.wanjari26@gmail.com'],mail_cc_list=[])      
        email_dataframes(df_dict_now , names_now ,'Strategy Performance' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])                       
        
    except:
        print('Error using emailer')
        


def fn_email_backtest_perf(Database , strat_names_2 , full_backtest_since , rebal_risk_daily):
    df_dict_now             = []
    names_now               = []           


############################################### Unit Risk Stats: YTD ##############################################################################################################
    df_unit                         = Database['Returns']['ALL IN Unit Returns'].copy()
    df_current_risk                 = Database['Returns']['ALL IN Current Risk Returns'].copy()
    df_optimized_risk               = Database['Returns']['ALL IN Optimized Returns'].copy()


    df_unit['COMPOSITE']            = df_unit.sum(axis = 1)
    df_current_risk['COMPOSITE']    = df_current_risk.sum(axis = 1)
    df_optimized_risk['COMPOSITE']  = df_optimized_risk.sum(axis = 1)
    
    df_unit                         = df_unit[strat_names_2]
    df_current_risk                 = df_current_risk[strat_names_2]
    df_optimized_risk               = df_optimized_risk[strat_names_2]

    df_unit.fillna(0, inplace = True)
    df_current_risk.fillna(0, inplace = True)
    df_optimized_risk.fillna(0, inplace = True)

############################################### Unit Risk Stats: YTD ##############################################################################################################
    
    current_year                    = str(date.today().year)
    df_7                            = df_unit.loc[current_year]
    df_7['CUMULATIVE']              = df_7['COMPOSITE'].cumsum()
    df_7                            = df_7.sort_index(ascending = False)
    df_7                            = df_7[['CUMULATIVE'] + strat_names_2]
    
    df_7_cum                        = (1+df_unit.loc[current_year]).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100

    df_7_cum['COMPOSITE'].plot()
    plt.title('Unit Risk: YTD Backtest - Cumulative Plot')
    plt.show()

    (mdd_percentage_all['COMPOSITE']/100).plot()
    plt.title('Unit Risk: YTD Backtest - Drawodown Plot')
    plt.show()

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD'  , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns =strat_names_2 )
    df_9.loc['Returns']             = df_8.sum()
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    df_9                            = round(df_9*100,3)
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.sum())/((df_8.std()*np.sqrt(252)))    
    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])

    df_dict_now.append(df_9) 
    names_now.append('Unit Risk Stats: YTD')  
    
############################################### Unit Risk Stats: FULL BACKTEST ##############################################################################################################
    
    df_7                            = df_unit.loc[full_backtest_since:]
    df_7_cum                        = (1+df_unit.loc[full_backtest_since:]).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100
    
    df_7_cum['COMPOSITE'].plot()
    plt.title('Unit Risk: Full Backtest - Cumulative Plot')
    plt.show()

    (mdd_percentage_all['COMPOSITE']/100).plot()
    plt.title('Unit Risk: Full Backtest - Drawodown Plot')
    plt.show()

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD'  , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns =strat_names_2 )
    df_9.loc['Returns']             = df_8.mean()*252
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    df_9                            = round(df_9*100,3)
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.mean()*252)/((df_8.std()*np.sqrt(252)))    
    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])

    df_dict_now.append(df_9) 
    names_now.append('Unit Risk Stats: FULL BACKTEST')  
    
    
############################################### Optimized Risk Stats: YTD ##############################################################################################################
    
    current_year                    = str(date.today().year)
    df_7                            = df_optimized_risk.loc[current_year]
    df_7['CUMULATIVE']              = df_7['COMPOSITE'].cumsum()
    df_7                            = df_7.sort_index(ascending = False)
    df_7                            = df_7[['CUMULATIVE'] + strat_names_2]
    
    df_7_cum                        = (1+df_optimized_risk.loc[current_year]).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100

    df_7_cum['COMPOSITE'].plot()
    plt.title('Optimized Risk: YTD Backtest - Cumulative Plot')
    plt.show()

    (mdd_percentage_all['COMPOSITE']/100).plot()
    plt.title('Optimized Risk: YTD Backtest - Drawodown Plot')
    plt.show()

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD'  , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns =strat_names_2 )
    df_9.loc['Returns']             = df_8.sum()
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    df_9                            = round(df_9*100,3)
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.sum())/((df_8.std()*np.sqrt(252)))    
    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])

    df_dict_now.append(df_9) 
    names_now.append('Optimized Risk Stats: YTD')  
    

    
############################################### Optimized Risk Stats: FULL BACKTEST ##############################################################################################################
    
    df_7                            = df_optimized_risk.loc[full_backtest_since:]
    df_7_cum                        = (1+df_optimized_risk.loc[full_backtest_since:]).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100
    
    df_7_cum['COMPOSITE'].plot()
    plt.title('Optimized Risk: Full Backtest - Cumulative Plot')
    plt.show()

    (mdd_percentage_all['COMPOSITE']/100).plot()
    plt.title('Optimized Risk: Full Backtest - Drawodown Plot')
    plt.show()

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD'  , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns =strat_names_2 )
    df_9.loc['Returns']             = df_8.mean()*252
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    df_9                            = round(df_9*100,3)
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.mean()*252)/((df_8.std()*np.sqrt(252)))    
    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])

    df_dict_now.append(df_9) 
    names_now.append('Optimized Risk Stats: FULL BACKTEST')  
        
    


    

############################################### Current Risk Stats: YTD ##############################################################################################################

    current_year                    = str(date.today().year)
    df_7                            = df_current_risk.loc[current_year]
    df_7['CUMULATIVE']              = df_7['COMPOSITE'].cumsum()
    df_7                            = df_7.sort_index(ascending = False)
    df_7                            = df_7[['CUMULATIVE'] + strat_names_2]
    
    df_7_cum                        = (1+df_current_risk.loc[current_year]).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100

    df_7_cum['COMPOSITE'].plot()
    plt.title('Current Risk: YTD Backtest - Cumulative Plot')
    plt.show()

    (mdd_percentage_all['COMPOSITE']/100).plot()
    plt.title('Current Risk: YTD Backtest - Drawodown Plot')
    plt.show()

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD'  , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns = strat_names_2 )
    df_9.loc['Returns']             = df_8.sum()
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    df_9 = round(df_9*100,3)
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.sum())/((df_8.std()*np.sqrt(252)))    
    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])

    df_dict_now.append(df_9) 
    names_now.append('Current Risk Stats: YTD') 

############################################### Current Risk Stats: FULL BACKTEST ##############################################################################################################

    df_7                            = df_current_risk.loc[full_backtest_since:]
    df_7_cum                        = (1+df_current_risk.loc[full_backtest_since:]).cumprod()
    hwm                             = df_7_cum.cummax()
    nav                             = df_7_cum
    mdd_percentage_all              = (nav/hwm - 1)*100

    df_7_cum['COMPOSITE'].plot()
    plt.title('Current Risk: Full Backtest - Cumulative Plot')
    plt.show()

    (mdd_percentage_all['COMPOSITE']/100).plot()
    plt.title('Current Risk: Full Backtest - Drawodown Plot')
    plt.show()

    df_8                            = df_7[strat_names_2]
    df_9                            = pd.DataFrame(data = 0 , index = ['Returns' , 'Vol', 'Sharpe', 'MDD', 'Current DD'  , 'Single D Loss'  , 'Single M Loss' ,'Single D Gain'  , 'Single M Gain'] , columns = strat_names_2 )
    df_9.loc['Returns']             = df_8.mean()*252
    df_9.loc['Vol']                 = df_8.std()*np.sqrt(252)
    df_9.loc['Single D Loss']       = df_8.min()
    df_9.loc['Single M Loss']       = df_8.resample('M').sum().min()
    df_9.loc['Single D Gain']       = df_8.max()
    df_9.loc['Single M Gain']       = df_8.resample('M').sum().max()
    df_9 = round(df_9*100,3)
    # df_9                            = integer_separator(df_9, df_9.columns)    
    df_9.loc['Sharpe']              = (df_8.mean()*252)/((df_8.std()*np.sqrt(252)))    
    df_9.loc['MDD']                 = (mdd_percentage_all).min()
    df_9.loc['Current DD']          = (mdd_percentage_all.iloc[-1])

    df_dict_now.append(df_9) 
    names_now.append('Current Risk Stats: FULL BACKTEST') 
    
############################################### UNIT Risk Stats: YEARLY BREAKDOWN ##############################################################################################################


    df_returns                      = df_unit['2007':].copy()  
    annual_return                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_vol                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_sharpe                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_mdd                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)

    for year in annual_sharpe.index:
        annual_return.loc[year]     = df_returns.loc[year].sum()*100
        annual_vol.loc[year]        = df_returns.loc[year].std()*np.sqrt(252)*100
        annual_sharpe.loc[year]     = (df_returns.loc[year].sum())/(df_returns.loc[year].std()*np.sqrt(252))
        df_cum                      = (1+df_returns.loc[year]).cumprod()
        df_HWM                      = df_cum.cummax()
        df_DD                       = df_cum/df_HWM - 1
        annual_mdd.loc[year]        = df_DD.min()*100


    df_dict_now.append(annual_return) 
    names_now.append('Unit Risk - Annual Realized Return')  
    df_dict_now.append(annual_vol) 
    names_now.append('Unit Risk - Annual Realized Vol') 
    df_dict_now.append(annual_sharpe) 
    names_now.append('Unit Risk - Annual Realized Sharpe') 
    df_dict_now.append(annual_mdd) 
    names_now.append('Unit Risk - Annual Realized MDD')     
    
############################################### Optimized Risk Stats: YEARLY BREAKDOWN ##############################################################################################################


    df_returns                      = df_optimized_risk['2007':].copy()  
    annual_return                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_vol                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_sharpe                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_mdd                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)

    for year in annual_sharpe.index:
        annual_return.loc[year]     = df_returns.loc[year].sum()*100
        annual_vol.loc[year]        = df_returns.loc[year].std()*np.sqrt(252)*100
        annual_sharpe.loc[year]     = (df_returns.loc[year].sum())/(df_returns.loc[year].std()*np.sqrt(252))
        df_cum                      = (1+df_returns.loc[year]).cumprod()
        df_HWM                      = df_cum.cummax()
        df_DD                       = df_cum/df_HWM - 1
        annual_mdd.loc[year]        = df_DD.min()*100



    df_dict_now.append(annual_return) 
    names_now.append('Optimized Risk - Annual Realized Return')  
    df_dict_now.append(annual_vol) 
    names_now.append('Optimized Risk - Annual Realized Vol') 
    df_dict_now.append(annual_sharpe) 
    names_now.append('Optimized Risk - Annual Realized Sharpe') 
    df_dict_now.append(annual_mdd) 
    names_now.append('Optimized Risk - Annual Realized MDD')     



    
############################################### CURRENT Risk Stats: YEARLY BREAKDOWN ##############################################################################################################
    
    df_returns                      = df_current_risk['2007':].copy()  
    annual_return                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_vol                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_sharpe                   = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)
    annual_mdd                      = pd.DataFrame(data = 0, index = [str(x) for x in list(df_returns['2007':].resample('Y').last().index.year)], columns = df_returns.columns)

    for year in annual_sharpe.index:
        annual_return.loc[year]     = df_returns.loc[year].sum()*100
        annual_vol.loc[year]        = df_returns.loc[year].std()*np.sqrt(252)*100
        annual_sharpe.loc[year]     = (df_returns.loc[year].sum())/(df_returns.loc[year].std()*np.sqrt(252))
        df_cum                      = (1+df_returns.loc[year]).cumprod()
        df_HWM                      = df_cum.cummax()
        df_DD                       = df_cum/df_HWM - 1
        annual_mdd.loc[year]        = df_DD.min()*100


    df_dict_now.append(annual_return) 
    names_now.append('Current Risk - Annual Realized Return')  
    df_dict_now.append(annual_vol) 
    names_now.append('Current Risk - Annual Realized Vol') 
    df_dict_now.append(annual_sharpe) 
    names_now.append('Current Risk - Annual Realized Sharpe') 
    df_dict_now.append(annual_mdd) 
    names_now.append('Current Risk - Annual Realized MDD')  
    
    try:
        # old_emailer.EmailRangeHandler(dfs_to_email = df_dict_now, dfs_header = names_now, outputdir=outputdir,outputfile_prefix = outfileprefix,mail_hdr='Strategy Backtest Performance',mail_list=['nishikant.wanjari26@gmail.com'],mail_cc_list=[])           
        email_dataframes(df_dict_now , names_now , 'Strategy Backtest Performance' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])

    except:
        print('Error using emailer')
        
    df_dict_now             = []
    names_now               = []         

    df_temnp_delta_fact             = df_optimized_risk.loc['2006':].resample('Y').sum()*250000000
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('Optimized Yearly P&L')  

    
    df_risk                         = rebal_risk_daily.loc['2019':].resample('BM').last()
    df_temnp_delta_fact             = df_optimized_risk.loc['2019':].resample('BM').sum()*250000000

### Multi index risk return    
    multi_index_df                          = pd.concat([df_risk, df_temnp_delta_fact], keys=['Risk', 'P&L'], axis=0)
    side_by_side_df                         = multi_index_df.unstack(0)
    side_by_side_df                         = side_by_side_df[strat_names_2]    
    side_by_side_df['COMPOSITE']['Risk']    = (df_optimized_risk['COMPOSITE'].rolling(window = 22).std()*np.sqrt(252)*100).reindex(side_by_side_df.index)
    df_dict_now.append(side_by_side_df) 
    names_now.append('Optimized Monthly P&L & Risk') 
     
        
    df_temnp_delta_fact             = df_optimized_risk.loc[current_year]*250000000
    temp_cols                       = list(df_temnp_delta_fact.columns)
    df_temnp_delta_fact['CUMULATIVE'] = df_temnp_delta_fact['COMPOSITE'].cumsum()
    df_temnp_delta_fact             = df_temnp_delta_fact[['CUMULATIVE'] + temp_cols]
    df_temnp_delta_fact             = df_temnp_delta_fact.sort_index(ascending = False)
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('Optimized Daily P&L: YTD')  

    email_dataframes(df_dict_now , names_now , 'Optimized P&L' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])
        
    
    return side_by_side_df


'''************************************************************************************************************************************'''

def fn_email_rolling_corr(df):
    df = Database['Returns']['ALL IN Unit Returns'].copy()
    
    df_dict_now             = []
    names_now               = []         

    df_temnp_delta_fact             = df.iloc[-66:].corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('3M Correlations')  
  
    df_temnp_delta_fact             = df.iloc[-132:].corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('6M Correlations')       

    df_temnp_delta_fact             = df.iloc[-198:].corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('9M Correlations')  
    
    df_temnp_delta_fact             = df.iloc[-264:].corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('1Y Correlations')  

    df_temnp_delta_fact             = df.iloc[-792:].corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('3Y Correlations')  

    df_temnp_delta_fact             = df.iloc[-1320:].corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('5Y Correlations')  

    df_temnp_delta_fact             = df.corr()
    df_dict_now.append(df_temnp_delta_fact) 
    names_now.append('FULL Correlations')      
    
    email_dataframes(df_dict_now , names_now , 'Rolling Unit Correlations' , ['nishikant.wanjari26@gmail.com'] , ['nishikant.wanjari26@gmail.com'])    
    



