# Heres-Waldo-Capstone-3
Utilizing Computer vision and Open CV to find waldo

Readme coming soon! Please wait!, Feel free to contact me with more information!

to create docker container, run these commands from the root of this github repo:  
**(Please note that this is very large(~3.5GB) due to the anaconda docker package being installed)**

>docker image build -t waldo_finder . 
   
>docker container run --publish 8080:8080 --detach --name waldo waldo_finder  
  
Then simply just go to the address listed in the terminal.(This will also work on an Could based instance like AWS EC2)