   10  curl --http1.1 'https://miniodis-rproxy.lisn.upsaclay.fr/coda-v2-prod-private/dataset/2025-02-09-1739091260/fe7abcfa6f6f/train_data_for_competition.zip?AWSAccessKeyId=codabench-prod&Signature=PzK1yqvo0Q%2BAjOxGDGgF7vkduqA%3D&Expires=1743007118' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Referer: https://www.codabench.org/' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: cross-site' -H 'Sec-Fetch-User: ?1' -H 'Priority: u=0, i' -H 'TE: trailers' -o data.zip
   11  unzip data
   12  sudo
   13  sudo apt-get install unzip
   14  unzip data
   15  sudo apt-get install git -y && git config --global user.name "tomtou-bspace" && git config --global user.email "tom.touati@brain.space"
   16  git clone git@github.com:Tom-Touati/mafat-challenge.git
   17  cd ssh
   18  cd .ssh
   19  ls
   20  echo """-----BEGIN OPENSSH PRIVATE KEY-----
   21  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
   68  -----END OPENSSH PRIVATE KEY-----""" > tom_git
   69  git clone git@github.com:Tom-Touati/mafat-challenge.git
   70  git config --global user.name "tomtou-bspace" && git config --global user.email "tom.touati@brain.space"
   71  git clone git@github.com:Tom-Touati/mafat-challenge.git
   72  git clone https://github.com/Tom-Touati/mafat-challenge.git
   73  mv mafat-challenge/ ..
   74  poetry
   75  poetry
   76  install python3.10 and poetry
   77  sudo apt install python3.10 && curl -sSL https://install.python-poetry.org | python3 -
   78  sudo apt-get update && sudo apt-get install python3.10
   79  curl -sSL https://install.python-poetry.org | python3
   80  echo 'export PATH="/home/tom.touati/.local/bin:$PATH"' >> ~/.bashrc
   81  exit
   82  ls
   83  exit
   84  free -h