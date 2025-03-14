import streamlit as st
import pandas as pd

st.set_page_config(page_title="Intelligent-System-project", layout="wide")

st.markdown('<h1 style="font-size: 40px;">🧠 Neural Network Deployment</h1>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

with st.expander("📌 **Neural Network คืออะไร!**"):
    st.info("""
    **Neural Network คือ** ระบบคอมพิวเตอร์จากโมเดลทางคณิตศาสตร์ ที่จำลองการทำงานของโครงข่ายประสาทชีวภาพในสมองของสัตว์  
    - โครงข่ายประสาทเทียมสามารถ **เรียนรู้** จากตัวอย่างโดยไม่ต้องถูกโปรแกรมด้วยกฎเกณฑ์ตายตัว  
    - ใช้ในงาน เช่น **การประมวลผลภาพ** เพื่อแยกแยะว่าเป็นแมวหรือไม่ โดยเรียนรู้จากชุดข้อมูลภาพแมวและไม่ใช่แมว  
    """)

with st.expander("📌 **ประโยชน์ของ Neural Network!**"):
    st.info("""
    - **ความสามารถในการเรียนรู้ที่ซับซ้อน**: Neural Network สามารถเรียนรู้รูปแบบที่ซับซ้อนในข้อมูลได้  
    - **การปรับปรุงประสิทธิภาพ**: Neural Network สามารถปรับปรุงประสิทธิภาพได้อย่างต่อเนื่อง  
    - **การสร้างนวัตกรรมใหม่ๆ**: เป็นรากฐานสำคัญในการพัฒนาเทคโนโลยี เช่น หุ่นยนต์อัจฉริยะ รถยนต์ไร้คนขับ และแพทยศาสตร์ที่แม่นยำยิ่งขึ้น  
    """)

st.write("<br>", unsafe_allow_html=True)

st.markdown('''
    <p style="font-size: 20px;">
        <a href="https://www.kaggle.com/datasets/samithsachidanandan/world-happiness-report-2020-2024" target="_blank" style="font-size: 25px; color: blue;">Kaggle.com</a>.
        <br>
        <a href="https://drive.google.com/file/d/15QptRjAFLRMnSaAu9Dsw46hYzk6ikHTt/view" target="_blank" style="font-size: 25px; color: red;">World Happiness Report (2020-2024) CSV File</a>.
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📚 เนื้อหาเกี่ยวกับ</h1><br>', unsafe_allow_html=True)
st.markdown('''
    <p style="font-size: 20px;">
        เป็นข้อมูลที่แสดงถึงดัชนีความสุขของโลกในปี พ.ศ. 2563 โดยแจกแจงตามประเทศต่างๆ ข้อมูลนี้รวบรวมโดย 
        Sustainable Development Solutions Network ซึ่งเป็นส่วนหนึ่งของโครงการริเริ่มของสหประชาชาติในการวัด
        ความสุขของประชากรโลก ข้อมูลนี้สามารถใช้เพื่อตรวจสอบแนวโน้มความสุขและเพื่อพัฒนากลยุทธ์ในการส่งเสริม
        ความเป็นอยู่ที่ดี
        ข้อสังเกตที่น่าสนใจบางประการจากข้อมูล:
            - ประเทศที่มีความสุขที่สุดในโลกคือฟินแลนด์ โดยมีคะแนนความสุข 7.81 คะแนน
            - ประเทศที่มีความสุขน้อยที่สุดในโลกคืออัฟกานิสถาน โดยมีคะแนนความสุข 2.57 คะแนน
            - ประเทศในแถบสแกนดิเนเวียมีแนวโน้มที่จะมีความสุขมากที่สุด
            - ประเทศในอนุภูมิภาคทะเลทรายซาฮาราในแอฟริกามีแนวโน้มที่จะมีความสุขน้อยที่สุด
            - มีความสัมพันธ์เชิงบวกที่แข็งแกร่งระหว่าง GDP ต่อหัวและความสุข
            - มีความสัมพันธ์เชิงบวกที่แข็งแกร่งระหว่างการสนับสนุนทางสังคมและความสุข
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📊 Features หลักๆ ที่มีอยู่ใน Dataset นี้</h1>', unsafe_allow_html=True)
with st.expander("📌 **Click Here to Learn More!**"):
    st.markdown("""
    - **Country name**: Name of the country.
    - **Happiness Rank**: Rank of the country
    - **Happiness Score**: Score of the country.
    - **Upperwhisker**: Upper score.
    - **Lowerwhisker**: Lower score.
    - **Economy (GDP per Capita)**: GDP.
    - **Social support**: Score from social support.
    - **Healthy life expectancy**: Score from Life Expectancy.
    - **Freedom to make life choices**: Score from Freedom.
    - **Generosity**: Score from Generosity.
    - **Perceptions of corruption**: Score from Perceptions of corruption.
    """, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">Show DataFrame As Dataset</h1>', unsafe_allow_html=True)
df = pd.read_csv("Dataset/2020.csv")  
st.dataframe(df)  

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🛠️ การเตรียมข้อมูล | พัฒนาโมเดล 
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('## Code Example')

code = '''
    # โหลดข้อมูลจากไฟล์ CSV (ต้องอัปโหลดไฟล์ไปยัง Google Colab ก่อน)
    from google.colab import files
    uploaded = files.upload()

    # ตรวจสอบข้อมูล
    df.head()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">เป็นการดูข้อมูล 5 แถวแรก หรือ header</h5>', unsafe_allow_html=True)
st.image(r"C:\Users\user\OneDrive - kmutnb.ac.th\รูปภาพ\สกรีนช็อต\13.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    df.isnull().sum()  # ตรวจสอบค่าที่หายไป
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ตรวจสอบดูว่ามี missing values หรือไม่</h5>', unsafe_allow_html=True)
st.image(r"C:\Users\user\OneDrive - kmutnb.ac.th\รูปภาพ\สกรีนช็อต\14.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # เลือกฟีเจอร์ที่ต้องการใช้
    features = ['Social support', 'Healthy life expectancy',
                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    target = 'Happiness score'

    X = df[features]
    y = df[target]

    # จัดการค่าหายไป
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    # แบ่งข้อมูลเป็น Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # ทำ Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
'''
st.code(code, language="python")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # สร้างและเทรน Neural Network โดยใช้ Regression Model
    mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='sgd',
                        learning_rate_init=0.01, max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้างและเทรนแบบจำลอง Neural Network เพื่อทำนายค่า "Happiness score" จาก features ที่กำหนด โดยใช้ Regression เป็นกลไกหลักในการปรับค่า weights ของแบบจำลอง</h5>', unsafe_allow_html=True)
st.image(r"C:\Users\user\OneDrive - kmutnb.ac.th\รูปภาพ\สกรีนช็อต\15.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # ทำนายค่าบนชุดทดสอบ
    y_pred = mlp.predict(X_test_scaled)
'''
st.code(code, language="python")

code = '''
    # คำนวณ Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ประเมินประสิทธิภาพของแบบจำลอง Neural Network โดยการคำนวณค่า MSE ซึ่งเป็นตัวชี้วัดว่าแบบจำลองทำนายค่า "Happiness score" ได้ใกล้เคียงกับค่าจริงมากน้อยเพียงใด ค่า MSE ที่ต่ำบ่งบอกถึงความแม่นยำของแบบจำลองที่สูงขึ้น</h5>', unsafe_allow_html=True)
st.image(r"C:\Users\user\OneDrive - kmutnb.ac.th\รูปภาพ\สกรีนช็อต\16.jpg")

st.write("<br>", unsafe_allow_html=True)

code = '''
    # แสดงผลลัพธ์เป็นกราฟ
    plt.figure(figsize=(8, 5))
    plt.plot(y_test.values, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', linestyle='dashed')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Happiness Score')
    plt.title('Neural Network Predictions vs True Values')
    plt.show()
'''
st.code(code, language="python")
st.image(r"C:\Users\user\OneDrive - kmutnb.ac.th\รูปภาพ\สกรีนช็อต\17.jpg")

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        📚 ทฤษฎีอัลกอริทึม
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
        <div>
            <h3 style="color: #FF5733;">1. พื้นฐานของ Regression Model</h3>
            <p style="color: #F8F8FF;">
                Regression Model เป็นเทคนิคที่ใช้คาดการณ์ค่าผลลัพธ์จากตัวแปรอินพุต ซึ่งสามารถแบ่งเป็นประเภทหลัก ๆ ได้แก่
            </p>
            <ul style="color: #F8F8FF;">
                <li><b>Linear Regression</b>: โมเดลที่ใช้สมการเชิงเส้น <br> y = Wx + b</li>
                <li><b>Polynomial Regression</b>: ขยายจาก Linear Regression โดยเพิ่มกำลังของตัวแปร x², x³, ...</li>
                <li><b>Multiple Regression</b>: มีมากกว่าหนึ่งตัวแปรอิสระ <br> y = w₁x₁ + w₂x₂ + ⋯ + b</li>
                <li><b>Neural Network Regression</b>: ใช้โครงข่ายประสาทเทียมที่มีหลายชั้นเพื่อทำการเรียนรู้ฟังก์ชันที่ซับซ้อน</li>
            </ul>
        </div>
    </div>
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
        <div>
            <h3 style="color: #FF5733;">2. Neural Network Regression Model</h3>
            <p style="color: #F8F8FF;">
                Neural Network Regression Model ใช้ Multilayer Perceptron (MLP) ในการทำ Regression โดยโครงสร้างหลักมีดังนี้
            </p>
            <ul style="color: #F8F8FF;">
                <li><b>Input Layer</b>: รับค่า features หรือคุณลักษณะของข้อมูล</li>
                <li><b>Hidden Layers</b>: ทำการแปลงค่าผ่าน weights (W) และ bias (b) ตาม Activation Function</li>
                <li><b>Output Layer</b>: ให้ค่าผลลัพธ์ต่อเนื่องออกมา</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #FFA07A;">🔹 ฟังก์ชันที่ใช้ใน Neural Network Regression</h4>
            <ul style="color: #F8F8FF;">
                <li><b>Activation Function:</b>
                    <ul>
                        <li>ReLU (Rectified Linear Unit): เหมาะสำหรับ hidden layer</li>
                        <li>Linear Activation (Identity Function): ใช้ที่ output layer</li>
                    </ul>
                </li>
                <li><b>Loss Function:</b>
                    <ul>
                        <li>Mean Squared Error (MSE): ใช้วัดค่าความผิดพลาด</li>
                        <li>Mean Absolute Error (MAE): วัดค่าความผิดพลาดโดยใช้ค่าสัมบูรณ์</li>
                    </ul>
                </li>
            </ul>
        </div>
        <div>
            <h4 style="color: #FFA07A;">🔹 ตัวอย่างสมการของ Neural Network Regression</h4>
            <p style="color: #F8F8FF; text-align: center;">
                y = W₃ ⋅ ReLU(W₂ ⋅ ReLU(W₁ X + b₁) + b₂) + b₃
            </p>
            <p style="color: #F8F8FF;">
                โดยที่:
                <ul>
                    <li><b>W</b> และ <b>b</b> คือ weights และ bias ของแต่ละชั้น</li>
                    <li><b>ReLU</b> คือ Activation Function ของ Hidden Layers</li>
                    <li>Output Layer ใช้ <b>Linear Activation</b></li>
                </ul>
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)






