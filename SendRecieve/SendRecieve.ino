int x = 0; // for incoming serial data
char y;


int SHARP1 = A1;   // sharp input pin
int sensor;       // sharp sensor reading

int stoppingBelt = 0;
int CheckIR = 5;
bool check;

int Motor = 7;

int servoPinL = 3;    //servo connected to digital pin 3
int servoPinR = 5;    //servo connected to digital pin 5


int LeftOffset = 15;
int RightOffset = 4;
int ZeroAngle = 90;
int LeftSideAngle = 45;
int RightSideAngle = 180-LeftSideAngle;

#include <Servo.h> //servo library call
Servo myservoL; // create myservo object for library Servo
Servo myservoR; // create myservo object for library Servo



void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); //set up serial buad rate (must be same as in python)
  Serial.setTimeout(1);

  pinMode(LED, OUTPUT); 
  pinMode(Motor, OUTPUT); 
  pinMode(servoPinL, OUTPUT); 
  pinMode(servoPinR, OUTPUT); 

  
  myservoL.attach(servoPinL);
  myservoR.attach(servoPinR);
  MiddleBin();

  delay(5000);

  
  digitalWrite(Motor, LOW);
  sensor = map(analogRead(SHARP1),0,1023,0,5000);

  if(Serial.available()){
        y = Serial.read();
  }
}



void loop() {
  // put your main code here, to run repeatedly:
  
  while (!Serial.available());
  x = Serial.readString().toInt();
  

  if(x==0){
    StopBelt();
    Serial.println(x);
    
  }else if(x==1){
    StartBelt();
    Serial.println(x);
    
  }else if(x==2){
    LeftBin();
    Serial.println(x);
    
  }else if (x==3){
    RightBin();
    Serial.println(x);

  }else if (x==4){
    MiddleBin();
    Serial.println(x);

  }else if (x==5){
    
    do{
      
      
      sensor = map(analogRead(SHARP1),0,1023,0,5000);
      
      //Serial.println(sensor);
      if (sensor >=2000){
        
        Serial.println(CheckIR);
        break;
      }
      delay(100);
      

    }while(true);

  }

}


//function to run while doing other tasks to see if we need to stop the belt for any reason
bool CheckStop(){
  if (Serial.available()){
      Serial.println("CheckingStop");
        x = Serial.readString().toInt();
        if (x==0){
          Serial.println("stoppingBelt");
          return true;
          
        }
      }else{
          return false;
        }
}


void StopBelt(){
  digitalWrite(Motor, LOW);
}

void StartBelt(){
  digitalWrite(Motor, HIGH);
}



void LeftBin(){
  myservoR.writeMicroseconds(AngleToPulse(ZeroAngle+RightOffset));
  delay(1000);
  myservoL.writeMicroseconds(AngleToPulse(LeftSideAngle+LeftOffset));
  delay(1000);
}

void MiddleBin(){
  myservoR.writeMicroseconds(AngleToPulse(ZeroAngle+RightOffset));
  delay(1000);
  myservoL.writeMicroseconds(AngleToPulse(ZeroAngle+LeftOffset));
  delay(1000);
}

void RightBin(){
  myservoL.writeMicroseconds(AngleToPulse(ZeroAngle+LeftOffset));
  delay(1000);
  
  myservoR.writeMicroseconds(AngleToPulse(RightSideAngle+RightOffset));
  delay(1000);
  
}


int AngleToPulse(int Angle){
  return map(Angle,0,180,500,2500);
}
