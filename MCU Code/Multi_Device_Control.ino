#include "FirebaseESP32.h"
#include <WiFi.h>


const char* ssid = "Fitters";
const char* password = "9.F1tt3r";

void connectWifi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(4500); //Time needed to open Serial Monitor manually, after powering ON the MCU
    Serial.print("Connecting to ");
    Serial.print(ssid);
    Serial.print("...");
    Serial.print("\n");
  }
  Serial.print("WiFi Connected to: ");
  Serial.print(ssid);
  Serial.print("\n");
  Serial.print("IP Address: ");
  Serial.print(WiFi.localIP());
  Serial.print("\n");
}

FirebaseData firebaseData;
void setup() {
  Serial.begin(9600);
  while (!Serial) {}
  connectWifi();
  Firebase.begin("https://ed-workshop.firebaseio.com/", "x9nP2fyHFzvkwigwuY9XmNkPRNvef3MQxT61K1kM");
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
  pinMode(14, OUTPUT);
  pinMode(15, OUTPUT);
  digitalWrite(12, HIGH);
  digitalWrite(13, HIGH);
  digitalWrite(14, HIGH);
  digitalWrite(15, HIGH);

}

int prev_val = 0;
void loop() {
  // put your main code here, to run repeatedly:
  if (Firebase.getInt(firebaseData, "/state/")) {
    if  (firebaseData.dataType() == "int") {
      int val = firebaseData.intData();
      int dev_no = val / 10;
      if (val != prev_val)
      {
        prev_val = val;
        Serial.print("State Updated. ");
        Serial.print("New value: ");
        Serial.print(val);
        Serial.print("\n");
      }
      int state = val % 10;
      digitalWrite(11 + dev_no, 1 - state);
    }
  }
}
