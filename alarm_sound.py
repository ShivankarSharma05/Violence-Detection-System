import pygame

# Initialize mixer safely
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def play_alarm():
    print("ALARM TRIGGERED")

    try:
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play(-1)   # loop in background
    except Exception as e:
        print("Sound error:", e)


def stop_alarm():
    pygame.mixer.music.stop()