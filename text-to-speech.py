#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Sample data (Years and corresponding values, e.g, states)
years=[[2015],[2016],[2017],[2018],[2019],[2020]]
values=[200,220,240,260,280,300]
#step:1 split the data into training and testing sets(80% training and 20% testing)
years_train,years_test,values_train,values_test=train_test_split(years,values,test_size=0.2,random_state=42)
#step:2 create and train the linear regression model on the training data
model=LinearRegression()
model.fit(years_train,values_train)
#Step:3 predict future value for the year 2025
future_year=[[2025]]
predicted_value=model.predict(future_year)
print(f"Predicted value for the year {future_year[0][0]}: {predicted_value[0]}")


# In[3]:


pip install SpeechRecognition pyttsx3 pyaudio


# In[2]:


import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """Function to convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def main():
    """Main function to run the bot."""
    speak("Hello! I am your simple bot from Malla Reddy College.")
    speak("You can say hello.")
   
    while True:
        command = input("You: ").lower()
       
        if "hello" in command:
            speak("Hi, welcome to Malla Reddy University!")
        elif "bye" in command or "exit" in command:
            speak("Goodbye! Have a great day!")
            break
        else:
            speak("I didn't understand that. Please say hello or bye.")

if __name__ == "__main__":
    main()


# In[ ]:




