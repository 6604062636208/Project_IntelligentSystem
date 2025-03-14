import streamlit as st
import pandas as pd

st.set_page_config(page_title="Intelligent-System-project", layout="wide")

st.sidebar.header("Navigation")
st.sidebar.page_link("Machine_Learning.py", icon="🤖", disabled=True)
st.sidebar.page_link("pages/Neural_Network.py", icon="🧠")
st.sidebar.page_link("pages/Demo_Machine_Learning.py", icon="📊")
st.sidebar.page_link("pages/Demo_Neural_Network.py", icon="📈")

st.markdown('<h1 style="font-size: 40px;">🤖 Machine Learning Deployment</h1>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

with st.expander("📌 **Machine Learning คืออะไร!**"):
    st.info("""
    **Machine Learning คือ** กระบวนการที่ทำให้คอมพิวเตอร์สามารถเรียนรู้และพัฒนาการทำงานให้ดีขึ้นเอง จากข้อมูลและสภาพแวดล้อมที่ได้รับ  
    - ไม่ต้องมีมนุษย์คอยกำกับหรือเขียนโปรแกรมใหม่เมื่อมีข้อมูลรูปแบบใหม่ ๆ  
    - คอมพิวเตอร์สามารถ **ตีความและตอบสนอง** ต่อข้อมูลได้เอง  
    - **ช่วยธุรกิจและอุตสาหกรรม** ในการวิเคราะห์ข้อมูล ลดต้นทุน และเพิ่มประสิทธิภาพในการแข่งขัน  
    """)

with st.expander("📌 **ประโยชน์ของ Machine Learning!**"):
    st.info("""
    **Machine Learning** สามารถนำมาใช้ทำประโยชน์ได้มากมาย ขึ้นอยู่กับจินตนาการของผู้พัฒนา  
    - **Google Maps**: ช่วยค้นหาเส้นทางที่ประหยัดเวลามากที่สุด  
    - **Google Translate**: นำ Automation มาทำงานร่วมกับ Machine Learning เพื่อช่วยแปลภาษาได้แม่นยำขึ้น  
    - **Speech-to-Text** (เช่น LINE Chat): ช่วยแปลงเสียงพูดเป็นข้อความ เพื่อลดเวลาการพิมพ์  
    """)

st.markdown('''
    <p style="font-size: 20px;">
        <a href="https://www.kaggle.com/datasets/sandeep1080/epidemiological-bmi-of-children-by-gender" 
           target="_blank" style="font-size: 25px; color: blue;">
           Kaggle.com
        </a>.
        <br>
        <a href="https://drive.google.com/file/d/1EWIfZ99J1-XDTkOBas6BDpMhJQxyBObk/view" 
           target="_blank" style="font-size: 25px; color: red;">
           Child Weight Categories CSV File
        </a>.
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📚 เนื้อหาเกี่ยวกับ</h1>', unsafe_allow_html=True)
st.markdown('''
    <p style="font-size: 20px;">
        เป็นข้อมูลที่แสดงดัชนีมวลกาย (BMI) ของนักเรียนในสกอตแลนด์ในช่วงปี พ.ศ. 2544 ถึง พ.ศ. 2566 โดย
        แจกแจงตามกลุ่มอายุ เพศ และพื้นที่ ข้อมูลนี้รวบรวมโดยโครงการ Scottish Health Survey ซึ่งเป็นส่วนหนึ่งของ
        โครงการริเริ่มระดับชาติของรัฐบาลสกอตแลนด์ในการตรวจสอบสุขภาพของเด็ก ข้อมูลนี้สามารถใช้เพื่อตรวจสอบ
        แนวโน้มของโรคอ้วนในเด็กและเพื่อพัฒนากลยุทธ์ในการส่งเสริมการมีน้ำหนักที่ดีต่อสุขภาพ
            
        ข้อสังเกตที่น่าสนใจบางประการจากข้อมูล:
            - อัตราโรคอ้วนในเด็กเพิ่มขึ้นในช่วงปี พ.ศ. 2544 ถึง พ.ศ. 2566
            - เด็กชายมีแนวโน้มที่จะเป็นโรคอ้วนมากกว่าเด็กหญิง
            - อัตราโรคอ้วนจะสูงกว่าในพื้นที่ด้อยโอกาส
            - อัตราโรคอ้วนจะสูงกว่าในเด็กโต
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📊 Features หลักๆ ที่มีอยู่ใน Dataset นี้</h1>', unsafe_allow_html=True)
with st.expander("📌 **Click Here to Learn More!**"):
    st.markdown("""
    - **SchoolYear**: The academic year;**Text**
    - **NameHospital**: Hospital or hospital board name;**Text**
    - **Sex**: The child's gender (Male/Female);**Text**
    - **EpiUnderweight**: Children with underweight BMI;**Number**
    - **EpiHealthyWeight**: Children with healthy weight BMI;**Number**
    - **EpiOverweight**: Children with overweight BMI;**Number**
    - **EpiObese**: Children with obesity;**Number**
    - **EpiOverweightAndObese**: Combined Children of overweight and obese;**Number**
    - **ValidCounts**: Valid child entries;**Number**
    - **UnvalidCounts**: Invalid child entries;**Number**
    - **TotalCounts**: Number of children;**Number**
    """, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🛠️ 1. การเตรียมข้อมูล
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="
        background-color: #000000; 
        padding: 20px; 
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin: 20px 0px;
    ">
        <p style="font-size: 20px; color: #F5FFFA; line-height: 1.8;">
            🔍 <b style="color: #FF5733;">การเตรียมข้อมูล</b> เป็นขั้นตอนที่สำคัญในการสร้างโมเดล  
            โดยในขั้นตอนนี้จะทำการเตรียมข้อมูลให้เหมาะสมสำหรับการสร้างโมเดล  
            ซึ่งประกอบไปด้วย 4 กระบวนการหลัก ได้แก่:
        </p>
        <div 
            <ul style="font-size: 18px; color: #F5FFFA; line-height: 1.6;">
                <li>✨ <b style="color: #FF5733;">การทำความสะอาดข้อมูล</b> - จัดการค่า Missing และค่าผิดปกติ</li>
                <li>📊 <b style="color: #FF5733;">การจัดกลุ่มข้อมูล</b> - ปรับโครงสร้างข้อมูลให้เป็นระเบียบ</li>
                <li>🔄 <b style="color: #FF5733;">การแปลงข้อมูล</b> - ปรับค่าข้อมูลให้อยู่ในรูปแบบที่เหมาะสม</li>
                <li>🎯 <b style="color: #FF5733;">การเลือกข้อมูลที่สำคัญ</b> - ตัดข้อมูลที่ไม่จำเป็นออก</li>
            </ul>
        </div>
    </div>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 35px; color: #FF5733;">
        📌 1.1 BMIData
    </h1>
""", unsafe_allow_html=True)

with st.expander("📌 **Click Here to Learn More!**"):
    st.markdown("""
        <h3 style="color: #FF5733;">📋 คอลัมน์ในข้อมูล BMIData</h3>
        <ul style="font-size: 18px; line-height: 1.6; color: #F5FFFA;">
            <li><b>SchoolYear</b> 🗓 - ปีการศึกษา</li>
            <li><b>NameHospital</b> 🏥 - ชื่อโรงพยาบาล</li>
            <li><b>Sex</b> 🚻 - เพศ (ชาย/หญิง)</li>
            <li><b>EpiUnderweight</b> ⚖ - จำนวนเด็กที่น้ำหนักต่ำกว่าเกณฑ์</li>
            <li><b>EpiHealthyWeight</b> 💪 - จำนวนเด็กที่มีน้ำหนักปกติ</li>
            <li><b>EpiOverweight</b> 🍔 - จำนวนเด็กที่มีน้ำหนักเกิน</li>
            <li><b>EpiObese</b> 🛑 - จำนวนเด็กที่เป็นโรคอ้วน</li>
            <li><b>TotalCounts</b> 🔢 - จำนวนเด็กทั้งหมด</li>
        </ul>

        <h3 style="color: #FF5733;">🛠 การเตรียมข้อมูล</h3>
        <ul style="font-size: 18px; color: #F5FFFA; line-height: 1.6;">
            <li>✅ เช็คค่า <b>Missing และ Outliers</b></li>
            <li>🔄 แปลงข้อมูลให้อยู่ในรูปที่เหมาะสม เช่น แปลง <b>Sex</b> ให้เป็นค่าตัวเลข</li>
            <li>📊 สร้าง <b>Features ใหม่</b> เช่น คำนวณ <b>เปอร์เซ็นต์เด็กอ้วน</b> ในแต่ละโรงพยาบาล</li>
        </ul>
    """, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">Show DataFrame As Dataset</h1>', unsafe_allow_html=True)
df = pd.read_csv("Dataset/BMIData.csv")  
st.dataframe(df)  

st.write("<br><br>", unsafe_allow_html=True)

st.markdown('## Code Example')
code = '''
    # ดูหัวของข้อมูล Dataset นี้
    df.head()
'''
st.code(code, language="python")

st.markdown('<h5 style="font-size: 20px;">เป็นการดูข้อมูล 5 แถวแรก หรือ header</h5>', unsafe_allow_html=True)
st.image("images/1.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # ตรวจสอบข้อมูลเบื้องต้น
    df.info()
    df.describe()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ใช้เพื่อดูโครงสร้างโดยรวมของข้อมูล และตรวจสอบชนิดข้อมูล</h5>', unsafe_allow_html=True)
st.image("images/2.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # ตรวจสอบ Missing Values หรือเปล่า
    df.isnull().sum()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ใช้สำหรับตรวจสอบว่ามีข้อมูลที่หายไป (Missing Values) ใน DataFrame หรือไม่ และถ้ามีจะมีจำนวนเท่าไหร่ในแต่ละคอลัมน์</h5>', unsafe_allow_html=True)
st.image("images/3.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # ตรวจสอบว่ามีข้อมูลที่ซ้ำกันหรือไม่
    df.duplicated().sum()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ใช้ตรวจสอบข้อมูลที่ซ้ำกันใน DataFrame ซึ่งเป็นขั้นตอนสำคัญในการเตรียมข้อมูลก่อนนำไปวิเคราะห์ เพื่อให้มั่นใจว่าข้อมูลมีความถูกต้องและพร้อมสำหรับการวิเคราะห์</h5>', unsafe_allow_html=True)
st.image("images/4.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # เราจะวิเคราะห์ข้อมูลเชิงสถิติเพื่อทำความเข้าใจการกระจายของข้อมูล
    sns.histplot(df['EpiOverweightAndObese'], kde=True)
    plt.show()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้าง Histogram เพื่อแสดงการกระจายของข้อมูลในคอลัมน์ EpiOverweightAndObese ซึ่งเป็นขั้นตอนสำคัญในการวิเคราะห์ข้อมูลเชิงสถิติ</h5>', unsafe_allow_html=True)
st.image("images/5.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # การสร้างฟีเจอร์ใหม่ (Feature Engineering)
    df['OverweightRatio'] = df['EpiOverweightAndObese'] / df['TotalCounts']
    # ตรวจสอบฟีเจอร์ใหม่
    df.head()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้างฟีเจอร์ใหม่ OverweightRatio โดยการคำนวณจากฟีเจอร์ที่มีอยู่เดิม ซึ่งเป็นเทคนิค Feature Engineering ที่ใช้เพื่อปรับปรุงข้อมูลให้เหมาะสมกับการวิเคราะห์และสร้างโมเดล Machine Learning มากขึ้น</h5>', unsafe_allow_html=True)
st.image("images/6.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # หากมีข้อมูลที่เป็น categorical เราอาจจะต้องทำการเข้ารหัสข้อมูลให้เป็น numerical
    df = pd.get_dummies(df, columns=['Sex', 'NameHospital'], drop_first=True)

    # ตรวจสอบข้อมูลหลังการเข้ารหัส
    df.head()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">แปลงข้อมูล Categorical ในคอลัมน์ Sex และ NameHospital ให้เป็น Numerical ด้วยเทคนิค One-Hot Encoding เพื่อเตรียมข้อมูลให้พร้อมสำหรับการสร้างโมเดล Machine Learning</h5>', unsafe_allow_html=True)
st.image("images/7.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # การแบ่งข้อมูลเป็น Train และ Test Set
    from sklearn.model_selection import train_test_split

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X = df.drop('EpiOverweightAndObese', axis=1)
    y = df['EpiOverweightAndObese']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ตรวจสอบขนาดของชุดข้อมูล
    print(X_train.shape, X_test.shape)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">แบ่งข้อมูลออกเป็น 2 ชุด คือ ชุดฝึก (Training Set) และ ชุดทดสอบ (Testing Set) ซึ่งเป็นขั้นตอนสำคัญในการสร้างและประเมินโมเดล Machine Learning</h5>', unsafe_allow_html=True)
st.image("images/8.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # การปรับขนาดข้อมูล (Scaling)
    # ปรับขนาดข้อมูลให้อยู่ในช่วงเดียวกันเพื่อให้โมเดลทำงานได้ดีขึ้น
    from sklearn.preprocessing import StandardScaler

    # ปรับขนาดข้อมูล
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ตรวจสอบข้อมูลหลังการปรับขนาด
    print(X_train[:5])
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ปรับขนาดข้อมูล (Data Scaling) โดยใช้เทคนิค Standardization ด้วย StandardScaler จากไลบรารี scikit-learn ในชุดฝึกและชุดทดสอบให้อยู่ในช่วงเดียวกันโดยใช้ Standardization ซึ่งเป็นเทคนิคที่นิยมใช้ในการเตรียมข้อมูลก่อนนำไปสร้างโมเดล Machine Learning</h5>', unsafe_allow_html=True)
st.image("images/9.jpg")

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        📚 2. ทฤษฎีอัลกอริทึม
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <div style="
        background-color: #1E1E1E; 
        padding: 25px; 
        border-radius: 12px;
        box-shadow: 3px 3px 12px rgba(255,255,255,0.2);
        margin: 20px 0px;
    ">
        <p style="font-size: 22px; color: #F8F8FF; line-height: 1.8; font-weight: bold;">
            การจำแนกกลุ่มน้ำหนักของเด็กด้วย <b>Machine Learning</b> <br>
            ใช้ <b>Random Forest</b> หรือ <b>Logistic Regression</b> เพื่อทำนาย 
            ว่าเด็กจะอยู่ในกลุ่ม <span style="color: lightgreen;">"Healthy"</span> หรือ  
            <span style="color: #FF6347;">"Overweight/Obese"</span>
        </p>
    <div 
        <h3 style="color: #FF5733;">1. Random Forest Algorithm</h3>
        <p style="color: #F8F8FF;">เป็นอัลกอริทึมแบบ <b>Ensemble Learning</b> ที่ใช้หลาย <b>Decision Trees</b> ทำงานร่วมกัน 
        โดยใช้การโหวตเสียงข้างมากเพื่อจำแนกข้อมูล ทำให้มีความแม่นยำสูงและลด Overfitting ได้ดี</p>
    </div>
    <div 
        <h4 style="color: #FFA07A;"> 🔹 กระบวนการทำงานของ Random Forest:</h4>
        <ul style="color: #F8F8FF;">
            <li><b>Bootstrap Sampling:</b> สุ่มตัวอย่างข้อมูลมาใช้สร้างต้นไม้หลายต้น</li>
            <li><b>Feature Selection:</b> เลือกตัวแปรบางส่วนสำหรับแต่ละต้นไม้</li>
            <li><b>Decision Tree Construction:</b> แต่ละต้นไม้เรียนรู้และตัดสินใจแยกกัน</li>
            <li><b>Majority Voting:</b> นำผลลัพธ์จากทุกต้นไม้มาทำการโหวตเสียงข้างมาก</li>
        </ul>
    </div>
    <div 
        <h3 style="color: #FF5733;">2. Logistic Regression</h3>
        <p style="color: #F8F8FF;">เป็นอัลกอริทึมสำหรับ <b>Binary Classification</b> โดยใช้สมการทางคณิตศาสตร์และ 
        <b>Sigmoid Function</b> เพื่อแปลงค่าผลลัพธ์ให้อยู่ในช่วง 0-1 และใช้ Threshold ในการแบ่งกลุ่มข้อมูล</p>
    </div>
    <div
        <h4 style="color: #FFA07A;"> 🔹 กระบวนการทำงานของ Logistic Regression:</h4>
        <ul style="color: #F8F8FF;">
            <li><b>คำนวณโอกาส:</b> ใช้สมการทางคณิตศาสตร์เพื่อหาโอกาสที่เด็กจะเป็น "Overweight/Obese"</li>
            <li><b>ใช้ Sigmoid Function:</b> แปลงค่าให้อยู่ระหว่าง 0 ถึง 1</li>
            <li><b>กำหนด Threshold:</b> ถ้าค่าความน่าจะเป็น > 0.5 จัดอยู่ในกลุ่ม "Overweight/Obese"</li>
        </ul>
    </div>
        <div 
            <h3 style="color: #FF5733;">🔹 เปรียบเทียบระหว่าง Random Forest และ Logistic Regression</h3>
            <table style="color: #F8F8FF; width: 100%; border-collapse: collapse; border: 1px solid white; text-align: center;">
                <tr style="background-color: #333;">
                    <th style="padding: 12px; border: 1px solid white;">เกณฑ์</th>
                    <th style="padding: 12px; border: 1px solid white;">Random Forest</th>
                    <th style="padding: 12px; border: 1px solid white;">Logistic Regression</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid white;">หลักการ</td>
                    <td style="padding: 10px; border: 1px solid white;">ใช้หลายต้นไม้ช่วยกันตัดสิน</td>
                    <td style="padding: 10px; border: 1px solid white;">ใช้สมการเชิงเส้น + Sigmoid Function</td>
                </tr>
                <tr style="background-color: #222;">
                    <td style="padding: 10px; border: 1px solid white;">ความแม่นยำ</td>
                    <td style="padding: 10px; border: 1px solid white;">สูง</td>
                    <td style="padding: 10px; border: 1px solid white;">ปานกลาง</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid white;">การตีความผลลัพธ์</td>
                    <td style="padding: 10px; border: 1px solid white;">ซับซ้อน</td>
                    <td style="padding: 10px; border: 1px solid white;">เข้าใจง่าย</td>
                </tr>
                <tr style="background-color: #222;">
                    <td style="padding: 10px; border: 1px solid white;">ความเร็ว</td>
                    <td style="padding: 10px; border: 1px solid white;">ช้ากว่า</td>
                    <td style="padding: 10px; border: 1px solid white;">เร็วกว่า</td>
                </tr>
            </table>
        </div>
    </div>
""", unsafe_allow_html=True)

st.write("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🚀 2.1 Machine Learning Model (ใช้กับ BMIData)
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color: #000000; padding: 15px; border-radius: 10px;">
        <h3 style="color: #F8F8FF;">🔍 อัลกอริทึมที่เลือก:</h3>
        <ul style="font-size: 18px; line-height: 1.6;">
            <li><b style="color: #e74c3c;">Random Forest</b> 🌲 <b style="color: #F8F8FF;">- ใช้หลายๆ Decision Trees รวมกันเพื่อลด Overfitting</b></li>
            <li><b style="color: #2ecc71;">Logistic Regression</b> ➗ <b style="color: #F8F8FF;">- เหมาะกับปัญหา Classification ที่มีแค่ 2 คลาส</b></li>
        </ul>
        <h3 style="color: #F8F8FF;">🎯 เป้าหมายของโมเดล:</h3>
        <p style="font-size: 18px; color: #F8F8FF;">
            ✅ ทำนายว่ากลุ่มเด็กแต่ละคนจะอยู่ในกลุ่ม 
            <b style="color: #27ae60;">"Healthy"</b> 🏃‍♂️ หรือ 
            <b style="color: #e67e22;">"Overweight/Obese"</b> 🍔
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        📚 3. ขั้นตอนการพัฒนาโมเดล
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🚀 3.1 Machine Learning (BMIData)
    </h1>
""", unsafe_allow_html=True)

with st.expander("📌 **ขั้นตอนการพัฒนาโมเดล!**"):
    st.markdown("""
        <div style="
            background-color: #1E1E1E; 
            padding: 25px; 
            border-radius: 12px;
            box-shadow: 3px 3px 12px rgba(255,255,255,0.2);
            margin: 20px 0px;
        ">
            <ul style="color: #F8F8FF; font-size: 18px; line-height: 1.6;">
                <li>📥 <b>โหลดข้อมูล</b> และทำ Data Preprocessing</li>
                <li>📊 <b>สร้าง Features ที่สำคัญ</b> เช่น เปอร์เซ็นต์เด็กอ้วน</li>
                <li>🤖 <b>เลือกโมเดล Machine Learning</b> (Random Forest / Logistic Regression)</li>
                <li>⚙️ <b>Train และ Evaluate โมเดล</b> โดยใช้ accuracy_score และ confusion_matrix</li>
                <li>💾 <b>บันทึกโมเดลลงไฟล์ pickle</b> และนำไปใช้ในแอป</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown('## Code Example')
code = '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # สร้างโมเดล Logistic Regression
    model = LogisticRegression()

    # ฝึกโมเดล
    model.fit(X_train, y_train)

    # ทำนายผลลัพธ์
    y_pred = model.predict(X_test)

    # วัดประสิทธิภาพของโมเดล
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้างโมเดล Logistic Regression เพื่อจำแนกค่า EpiOverweightAndObese ออกเป็นกลุ่ม จากนั้นฝึกฝนโมเดลด้วยข้อมูลที่เตรียมไว้ และประเมินประสิทธิภาพของโมเดลโดยใช้ค่า Accuracy Score.</h5>', unsafe_allow_html=True)

code = '''
    from sklearn.ensemble import RandomForestRegressor

    # สร้างโมเดล Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # ฝึกโมเดล
    rf_model.fit(X_train, y_train)

    # ทำนายผลลัพธ์
    y_pred_rf = rf_model.predict(X_test)

    # วัดประสิทธิภาพของโมเดล
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print(f'Random Forest Mean Squared Error: {mse_rf}')
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้างโมเดล Random Forest Regressor เพื่อทำนายค่า EpiOverweightAndObese จากนั้นฝึกฝนโมเดลด้วยข้อมูลที่เตรียมไว้ และประเมินประสิทธิภาพของโมเดลโดยใช้ค่า Mean Squared Error.</h5>', unsafe_allow_html=True)

code = '''
    # คำนวณ Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # แสดง Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ประเมินประสิทธิภาพของโมเดล Classification โดยใช้ Accuracy และ Confusion Matrix</h5>', unsafe_allow_html=True)