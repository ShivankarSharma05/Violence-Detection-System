import time

last_sos_time = 0
cooldown = 30

def send_sos():
    global last_sos_time

    if time.time() - last_sos_time > cooldown:
        print("🚨 SOS SENT TO POLICE 🚨")
        last_sos_time = time.time()
    else:
        print("Cooldown active...")