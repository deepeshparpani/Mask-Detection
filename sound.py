import gtts  
from playsound import playsound  
t1 = gtts.gTTS("Please wear a mask")
t1.save("alert.mp3")     
playsound("alert.mp3")  
