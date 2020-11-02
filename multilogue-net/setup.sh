# clone git repo
rm -rf multilogue-net
git clone https://github.com/amanshenoy/multilogue-net
cd multilogue-net

# install dependencies
sed -i "s/mkl-fft==1.0.15/mkl-fft/" requirements.txt
sed -i "s/mkl-service==2.3.0/mkl/" requirements.txt
sed -i "s/PyYAML==5.1.2/PyYAML/" requirements.txt
pip install -r requirements.txt

# get data
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs fetch
git checkout 5d6b6ff8b1a26cf0762d6c1ca3a99917e881bf26 data/

