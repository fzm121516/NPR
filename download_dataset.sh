
pwd=$(cd $(dirname $0); pwd)
echo pwd: $pwd

# pip install gdown==4.7.1 

mkdir dataset
cd dataset

# --proxy http://ip:port



# https://github.com/Yuheng-Li/UniversalFakeDetect
# https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-
gdown https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO- -O ./UniversalFakeDetect --folder --continue
cd ./UniversalFakeDetect
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd $pwd/dataset

# https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
# https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing
gdown https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj -O ./GANGen-Detection --folder --continue

cd ./GANGen-Detection
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd $pwd/dataset

# https://github.com/ZhendongWang6/DIRE
# https://drive.google.com/drive/folders/1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD?usp=sharing
gdown https://drive.google.com/drive/folders/1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD -O ./DiffusionForensics --folder --continue

cd ./DiffusionForensics
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd $pwd/dataset

# https://github.com/Ekko-zn/AIGCDetectBenchmark
# https://drive.google.com/drive/folders/1BUv1MT1cm90QN3WTMHLEr8PXBsKGxKC9?usp=sharing
gdown https://drive.google.com/drive/folders/1BUv1MT1cm90QN3WTMHLEr8PXBsKGxKC9 -O ./AIGCDetect_testset --folder
zip -s- test.zip -O test_full.zip
unzip test_full.zip -d ./AIGCDetect_testset
cd $pwd/dataset

gdown https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq -O ./Diffusion1kStep --folder --continue
cd ./Diffusion1kStep
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd $pwd/dataset


# https://github.com/peterwang512/CNNDetection
gdown https://drive.google.com/uc?id=1AhWOsdCalrXE_6RmBZzyiC1ehXa6GXBK -O CNN_synth_testset.zip --continue
tar -zxvf CNN_synth_testset.zip -C ./ForenSynths

