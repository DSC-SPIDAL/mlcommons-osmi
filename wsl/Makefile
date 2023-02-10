PROJECT=/home/$(USER)/project/osmi
USER_NAME=`whoami`
GROUP_NAME=`id -g`
USER_ID=`id -u`
GROUP_ID=`getent group | fgrep ${GROUP_NAME} | cut -d":" -f 3`

image:
	time docker pull tensorflow/serving

run:
	docker run --rm --name osmi \
		-u $(USER_ID):$(GROUP_NAME) \
		-v $(PROJECT)/osmi-bench/models/small_lstm:/models/model \
		-p 8500:8500 \
		-p 8501:8501 \
		-d \
		tensorflow/serving

train:
	cd $(PROJECT)/osmi-bench/models; python train.py small_lstm

shell:
	docker exec -it osmi bash
  
kill:
	docker kill osmi

ps:
	docker ps
