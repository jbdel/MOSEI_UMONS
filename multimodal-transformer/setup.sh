# clone repo
rm -rf Multimodal-Transformer 
git clone https://github.com/yaohungt/Multimodal-Transformer.git
cd Multimodal-Transformer

# fetch data
mkdir data pre_trained_models
cd data
# manually get data from https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0
# wget may cause the download file unable to be unzip on some machines, though it works on colab
