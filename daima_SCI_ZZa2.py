# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:52:06 2024

@author: lenovo
"""



import streamlit as st

import pandas as pd

import joblib

import shap as shap  

import streamlit.components.v1 as components

#加载机器学习模型

model=joblib.load('./model_xgboost_zza.pkl')

#创建应用程序页面

st.title('机器学习模型预测')

st.write('请填写以下表单进行预测：')

#创建表单并收集用户输入

D6=st.number_input('新生儿体重(2100-5000克)：',min_value=2100,max_value=5000)
if D6<2100 or D6>5000:
   st.write('数据超出范围')

D4=st.number_input('第二产程时间(10-360分钟)：',min_value=10,max_value=360)
if D4<10 or D4>360:
   st.write('数据超出范围')

D31=st.number_input('第一产程潜伏期时间(20-1440分钟)：',min_value=20,max_value=1440)
if D31<20 or D31>1440:
   st.write('数据超出范围')

D32=st.number_input('第一产程活跃期时间(10-720分钟)：',min_value=10,max_value=720)
if D32<10 or D32>720:
   st.write('数据超出范围')

A6=st.number_input('BMI(17.0-35.9)：',min_value=17.0,max_value=35.9)
if A6<17.0 or A6>35.9:
   st.write('数据超出范围')

D5=st.number_input('第三产程时间(1-120分钟)：',min_value=1,max_value=120)
if D5<1 or D5>120:
   st.write('数据超出范围')

A1=st.number_input('年龄(18-50)：',min_value=18,max_value=50)
if A1<18 or A1>50:
   st.write('数据超出范围')

C1=st.number_input('孕周(37-43)：',min_value=37,max_value=43)
if C1<37 or C1>43:
   st.write('数据超出范围')

D7=st.number_input('新生儿身长(43-60)：',min_value=43,max_value=60)
if D7<43 or D7>60:
   st.write('数据超出范围')


##分类变量
A4_1=st.selectbox('职业：',['无','低体力','中体力','高体力'])
C9_1=st.selectbox('胎膜早破：',['否','有'])
C4_1=st.selectbox('贫血：',['否','有'])
D1_1=st.selectbox('分娩时刻：',['0 o_clock','1 o_clock','2 o_clock','3 o_clock','4 o_clock','5 o_clock','6 o_clock','7 o_clock',
                           '8 o_clock','9 o_clock','10 o_clock','11 o_clock','12 o_clock','13 o_clock','14 o_clock','15 o_clock',
                          '16 o_clock','17 o_clock','18 o_clock','19 o_clock','20 o_clock','21 o_clock','22 o_clock','23 o_clock'])
C3_1=st.selectbox('高血压：',['否','有'])
D8_1=st.selectbox('分娩镇痛：',['否','有'])
###分类变量变数值
if A4_1 == "无":
    A4 = 0
elif A4_1 == "低体力":
    A4 = 1
elif A4_1 == "中体力":
    A4 = 2
elif A4_1 == "高体力":
    A4 = 3
else:
    A4 = 50000
    
if C9_1 == "否":
    C9 = 0
elif C9_1 == "有":
    C9 = 1
else:
    C9 = 50000
    
if C4_1 == "否":
    C4 = 0
elif C4_1 == "有":
    C4 = 1
else:
    C4 = 50000

if D1_1== "0 o_clock":
    D1=0
elif D1_1== "1 o_clock":
    D1 = 1
elif D1_1== "2 o_clock":
    D1 = 2
elif D1_1== "3 o_clock":
    D1 = 3
elif D1_1== "4 o_clock":
    D1 = 4
elif D1_1== "5 o_clock":
    D1 = 5
elif D1_1== "6 o_clock":
    D1 = 6
elif D1_1== "7 o_clock":
    D1 = 7
elif D1_1== "8 o_clock":
    D1 = 8
elif D1_1== "9 o_clock":
    D1 = 9
elif D1_1== "10 o_clock":
    D1 = 10
elif D1_1== "11 o_clock":
    D1 = 11
elif D1_1== "12 o_clock":
    D1 = 12
elif D1_1== "13 o_clock":
    D1 = 13
elif D1_1== "14 o_clock":
    D1 = 14
elif D1_1== "15 o_clock":
    D1 = 15
elif D1_1== "16 o_clock":
    D1 = 16
elif D1_1== "17 o_clock":
    D1 = 17
elif D1_1== "18 o_clock":
    D1 = 18
elif D1_1== "19 o_clock":
    D1 = 19
elif D1_1== "20 o_clock":
    D1 = 20
elif D1_1== "21 o_clock":
    D1 = 21
elif D1_1== "22 o_clock":
    D1 = 22
elif D1_1== "23 o_clock":
    D1 = 23
else:
    D1=50000

    
if C3_1 == "否":
    C3 = 0
elif C3_1 == "有":
    C3 = 1
else:
    C3 = 50000
    
if D8_1 == "否":
    D8 = 0
elif D8_1 == "有":
    D8 = 1
else:
    D8 = 50000

#进行预测并显示结果

if st.button('预测结果'):
    if A4==50000 or C9==50000 or C4==50000 or D1==50000 or C3==50000 or D8==50000:
        st.write('请填写数据')          
    else:
        data0=pd.DataFrame({'D6':[D6],'D31':[D31],'D4':[D4],'D32':[D32],'A4':[A4],'C9':[C9],'A6':[A6],
                           'C4':[C4],'D5':[D5],'A1':[A1],'D1':[D1],'D8':[D8],'C1':[C1],'D7':[D7],'C3':[C3]})
        features = ["A4","C9","C4","D1", "C3", "D8"]
        data2 = pd.get_dummies(data0[features])
        data1=data0[['D6','D31','D32','D4','A6','D5','A1','C1','D7']]
        data3=pd.concat([data1,data2],axis=1) 
        data=data3[['D6','D31','D4','D32','A4','C9','A6','C4','D5','A1','D1','D8','C1','D7','C3']]
        prediction=model.predict_proba(data)[:,1][0]
        def to_percentage(num):
            return '{:.2f}%'.format(num * 100)
        prediction1=to_percentage(prediction)
        explainer = shap.TreeExplainer(model)
        shap.initjs()   #初始化JS
        shap_values = explainer.shap_values(data) 
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)
        if prediction >0.13193563:
            st.write(f"发生产出血的概率为：{prediction1}，发生产后出血风险高")          
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], data.iloc[0,:]),400)
        else:
            st.write(f"发生产出血的概率为：{prediction1}，发生产后出血风险低")  
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], data.iloc[0,:]),400)

      
 