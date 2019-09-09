int c;
int RED = 3;
int GREEN = 5;
int BLUE = 6;
int IN1 = 11;
int IN2 = 10;
int IN3 = 9;
int IN4 = 8;

void setup() {
  Serial.begin(9600);
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(BLUE, OUTPUT);
  pinMode(IN1, OUTPUT);  
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);  
  pinMode(IN4, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    c = Serial.read() - 48;
    Serial.print(c);
    
    if(c == 0){
      digitalWrite(RED, HIGH);
      digitalWrite(GREEN, LOW);
      digitalWrite(BLUE, LOW);

      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
    }
    
    if(c == 1){
      digitalWrite(RED, LOW);
      digitalWrite(GREEN, HIGH);
      digitalWrite(BLUE, LOW);
      
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
    }
    
    if(c == 2){
      digitalWrite(RED, LOW);
      digitalWrite(GREEN, LOW);
      digitalWrite(BLUE, HIGH);

      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
    }
  }
}
