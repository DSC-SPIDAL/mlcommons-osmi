run:
  docker run --rm --name osmi -v ~/School/Research/osmi/osmi-bench/models/small_lstm:/models/model -p 8500:8500 -d tensorflow/serving

shell:
  docker exec -it tensorflow/serving /bin/bash
  
 kill:
  docker kill $(docker ps -q -a)
