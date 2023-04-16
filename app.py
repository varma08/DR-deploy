import streamlit as st
import time
import requests
import cv2
import webbrowser
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

st.set_page_config(layout = "wide")

page = st.sidebar.selectbox('Select page',
  ['Home page','Predictor','Q&A','Infographics'])

def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

if page == 'Home page':
    left_column,middle_column,right_column = st.columns(3)
    with middle_column:
      st.title("Diabetic Retinopathy Project")
    lottie_coding_eye = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_xa0q7ly3.json")
    
    with st.container():
        st.write("---")
        with middle_column:
            
            st_lottie(lottie_coding_eye, height=300, key="coding_eye")
    
    with left_column:
      with st.container():
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.title("PREDICTOR")
        st.write("A custom CNN model trained with over 3500 images gives predictions on the uploaded retinal image of any size if it is healthy or has a diabetic retinopathy condition with approximately an accuracy of 95%")
        lottie_coding_mini = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_KS2VTJka6L.json")
        st_lottie(lottie_coding_mini, height=200, key="brain")
        
    with middle_column:
      with st.container():
        st.write(" ")
        st.title("FAQs")
        st.write("Answers to all the frequently asked questions on the topic of diabetic retinopathy have been diligently curated from verified healthcare sources.")
        lottie_coding_tabs = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_au4zdsr8.json")
        st_lottie(lottie_coding_tabs, height=200, key="text")
    with right_column:
      with st.container():
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.title("INFOGRAPHICS")
        st.write("Some crucial statistical plots have been collected to give the user an idea of how severe and widescale this condition is on a national and global scale.")
        lottie_coding_stats = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_r7h02cq4.json")
        st_lottie(lottie_coding_stats, height=200, key="stat")
if page == 'Predictor':
                st.title("Diabetic Retinopathy Predictor")
                lottie_coding_pred = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_wusrjror.json")
                left_column,middle_column,right_column = st.columns(3)
                with middle_column:
                    st_lottie(lottie_coding_pred, height=300, key="pred")
                if "uploaded_file" not in st.session_state:
                    st.session_state["uploaded_file"] = ""
                model = tf.keras.models.load_model("model_weights.hdf5")
                uploaded_file = st.file_uploader("Upload an image")
                
                map_dict = {0: 'DR',
                            1: 'No_DR'
                            }

                st.session_state["uploaded_file"] = uploaded_file
                
                if uploaded_file is not None:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(opencv_image,(224,224))
                        # Now do something with the image! For example, let's display it:
                    st.image(opencv_image, channels="RGB")
                    st.success("Photo uploaded successfully !!!")

                    resized = mobilenet_v2_preprocess_input(resized)
                    img_reshape = resized[np.newaxis,...]

                    Genrate_pred = st.button("Generate Prediction",key="actual_pred")
                    if Genrate_pred:
                              prediction = model.predict(img_reshape).argmax()
                              st.title("Predicted Label for the image is {}".format(map_dict[prediction]))

if page == 'Q&A':
      left_column,right_column = st.columns(2)
      
      with left_column:
          st.title(" Q&A Section ")
          with st.expander("What is diabetic retinopathy?"):
              st.write("Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people who have diabetes. It affects blood vessels in the retina (the light-sensitive layer of tissue in the back of your eye. If you have diabetes, it’s important to get a comprehensive dilated eye exam at least once a year. Diabetic retinopathy may not have any symptoms at first — but finding it early can help you take steps to protect your vision. ")
              st.video("https://youtu.be/JxMOsMDM1UM")
          with st.expander("What are the symptoms of diabetic retinopathy?"):
              st.write("The early stages of diabetic retinopathy usually don’t have any symptoms. Some people notice changes in their vision, like trouble reading or seeing faraway objects. These changes may come and go. In later stages of the disease, blood vessels in the retina start to bleed into the vitreous (gel-like fluid that fills your eye). If this happens, you may see dark, floating spots or streaks that look like cobwebs. Sometimes, the spots clear up on their own — but it’s important to get treatment right away. Without treatment, scars can form in the back of the eye. Blood vessels may also start to bleed again, or the bleeding may get worse. ")
          with st.expander("Am I at risk for diabetic retinopathy?"):
              st.write("Anyone with any kind of diabetes can get diabetic retinopathy — including people with type 1, type 2, and gestational diabetes (a type of diabetes that can develop during pregnancy. Your risk increases the longer you have diabetes. Over time, more than half of people with diabetes will develop diabetic retinopathy. The good news is that you can lower your risk of developing diabetic retinopathy by controlling your diabetes. Women with diabetes who become pregnant — or women who develop gestational diabetes — are at high risk for getting diabetic retinopathy. If you have diabetes and are pregnant, have a comprehensive dilated eye exam as soon as possible. Ask your doctor if you’ll need additional eye exams during your pregnancy. ")
          with st.expander("What causes diabetic retinopathy?"):
              st.write("Diabetic retinopathy is caused by high blood sugar due to diabetes. Over time, having too much sugar in your blood can damage your retina — the part of your eye that detects light and sends signals to your brain through a nerve in the back of your eye. Diabetes damages blood vessels all over the body. The damage to your eyes starts when sugar blocks the tiny blood vessels that go to your retina, causing them to leak fluid or bleed. To make up for these blocked blood vessels, your eyes then grow new blood vessels that don’t work well. These new blood vessels can leak or bleed easily. ")
          with st.expander("How will my eye doctor check for diabetic retinopathy?"):
              st.write("Eye doctors can check for diabetic retinopathy as part of a dilated eye exam. The exam is simple and painless — your doctor will give you some eye drops to dilate (widen) your pupil and then check your eyes for diabetic retinopathy and other eye problems. If your eye doctor thinks you may have severe diabetic retinopathy or DME, they may do a test called a fluorescein angiogram. This test lets the doctor see pictures of the blood vessels in your retina.")
          with st.expander("What can I do to prevent diabetic retinopathy?"):
              st.write("Managing your diabetes is the best way to lower your risk of diabetic retinopathy. That means keeping your blood sugar levels in a healthy range. You can do this by getting regular physical activity, eating healthy, and carefully following your doctor’s instructions for your insulin or other diabetes medicines. To make sure your diabetes treatment plan is working, you’ll need a special lab test called an A1C test. This test shows your average blood sugar level over the past 3 months. You can work with your doctor to set a personal A1C goal. Meeting your A1C goal can help prevent or manage diabetic retinopathy. Having high blood pressure or high cholesterol along with diabetes increases your risk for diabetic retinopathy. So controlling your blood pressure and cholesterol can also help lower your risk for vision loss.")
          with st.expander("What’s the treatment for diabetic retinopathy and DME?"):
              st.write("In the early stages of diabetic retinopathy, your eye doctor will probably just keep track of how your eyes are doing. Some people with diabetic retinopathy may need a comprehensive dilated eye exam as often as every 2 to 4 months.")
              st.write("In later stages, it’s important to start treatment right away — especially if you have changes in your vision. While it won’t undo any damage to your vision, treatment can stop your vision from getting worse. It’s also important to take steps to control your diabetes, blood pressure, and cholesterol.")
              st.write("Injections : Medicines called anti-VEGF drugs can slow down or reverse diabetic retinopathy. Other medicines, called corticosteroids, can also help.")
              st.write("Laser treatment. To reduce swelling in your retina, eye doctors can use lasers to make the blood vessels shrink and stop leaking.")
              st.write("Eye surgery. If your retina is bleeding a lot or you have a lot of scars in your eye, your eye doctor may recommend a type of surgery called a vitrectomy.")
          with st.expander("How Are Eye Injections Done?"):
              st.video("https://youtu.be/esZTnQQpJTo")
          with st.expander("What is Vitrectomy Surgery?"):
              st.video("https://youtu.be/5-XY-_AwBMs")
          with st.expander("What is the latest research on diabetic retinopathy and DME?"):
              st.write("Scientists are studying better ways to find, treat, and prevent vision loss in people with diabetes. One NIH-funded research team is studying whether a cholesterol medicine called fenofibrate can stop diabetic retinopathy from getting worse.")
          
      with right_column:
    
          lottie_coding_qna = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_zntl98s1.json")
          st_lottie(lottie_coding_qna, height=700, key="qna")

if page == 'Infographics':
      st.title(" Infographics ")
      left_column,right_column = st.columns(2)
     
      with left_column:
        with st.expander("Points to keep in mind"):
          st.image("/content/Screenshot 2023-04-07 at 3.34.54 PM 2.png")
        with st.expander("Global Diabetic Retinopathy Market Share"):
          st.image("/content/Marketshare.jpeg")
        with st.expander("Global prevalence of Diabetic Retinopathy"):
          st.image("/content/Globalprevalence.jpeg")
        with st.expander("Global Diabetic Retinopathy Drugs Market"):
          st.image("/content/Drugsmarket.jpeg")
          
      with right_column:
          lottie_coding_info = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_22mjkcbb.json")
          st_lottie(lottie_coding_info, height=500, key="info")

