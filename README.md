# finefood
Amazon Fine Food and Polyaxon

Dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews/version/2

GloVe: http://nlp.stanford.edu/data/glove.6B.zip

Downloaded data expected in:

```
./data/Reviews.csv
./data/glove.6B.100d.txt
```

# References

https://docs.polyaxon.com/

https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html

# Local runs

Models can be trained local via command line:

```
python finefood/run.py --num_epochs=10 --batch_size=32 --sample_size=10000 --max_len=1000 --learning_rate=0.01 --dropout=0.5 --model_type=cnn_lstm
```

# Implemented models

## 2 layer lstm

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 1000)              0
_________________________________________________________________
embedding_1 (Embedding)      (None, 1000, 100)         2000000
_________________________________________________________________
lstm_1 (LSTM)                (None, 1000, 128)         117248
_________________________________________________________________
dropout_1 (Dropout)          (None, 1000, 128)         0
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               131584
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 645
_________________________________________________________________
activation_1 (Activation)    (None, 5)                 0
=================================================================
Total params: 2,249,477
Trainable params: 249,477
Non-trainable params: 2,000,000
```


## 1d cnn & lstm

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 1000)              0
_________________________________________________________________
embedding_1 (Embedding)      (None, 1000, 100)         2000000
_________________________________________________________________
dropout_1 (Dropout)          (None, 1000, 100)         0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 996, 128)          64128
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 249, 128)          0
_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               131584
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 645
_________________________________________________________________
activation_1 (Activation)    (None, 5)                 0
=================================================================
Total params: 2,196,357
Trainable params: 196,357
Non-trainable params: 2,000,000
```


## 3 layer 1d cnn

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 1000)              0
_________________________________________________________________
embedding_1 (Embedding)      (None, 1000, 100)         2000000
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 996, 128)          64128
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 199, 128)          0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 195, 128)          82048
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 39, 128)           0
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 35, 128)           82048
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 1, 128)            0
_________________________________________________________________
flatten_1 (Flatten)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645
=================================================================
Total params: 2,245,381
Trainable params: 245,381
Non-trainable params: 2,000,000
```



# Polyaxon setup

## Create a Single Node Filer:

The [click-to-deploy single-node file server](https://console.cloud.google.com/launcher/details/click-to-deploy-images/singlefs) provides a ZFS file server running on a single Google Compute Engine instance.

You need to create a filer: `polyaxon-nfs`, and keep the default value `data`, and check `enable NFS sharing`. You can set the storage to 50GB for example.

## Use ssh to create some folders for data, logs, outputs, upload, and repos under /data :

```
gcloud --project "amazonfinefood" compute ssh --ssh-flag=-L3000:localhost:3000 --zone=us-central1-b polyaxon-nfs-vm\
```

```
cd /data
mkdir -m 777 data
mkdir -m 777 outputs
mkdir -m 777 logs
mkdir -m 777 repos
mkdir -m 777 upload
```

## Get the ip address of the filer:

```
gcloud --project "amazonfinefood" compute instances describe polyaxon-nfs-vm --zone=us-central1-b --format='value(networkInterfaces[0].networkIP)'
```

```
$ vi gke/data-pvc.yml
# And replace with the right ip address

$ vi gke/outputs-pvc.yml
# And replace with the right ip address

$ vi gke/logs-pvc.yml
# And replace with the right ip address

$ vi gke/repos-pvc.yml
# And replace with the right ip address

$ vi gke/upload-pvc.yml
# And replace with the right ip address
```

## Use kubectl to create a namespace polyaxon
Use kubectl to create the PVCs based on the nfs server

```
kubectl create namespace polyaxon
kubectl create -f gke/data-pvc.yml -n polyaxon
kubectl create -f gke/outputs-pvc.yml -n polyaxon
kubectl create -f gke/upload-pvc.yml -n polyaxon
kubectl create -f gke/logs-pvc.yml -n polyaxon
kubectl create -f gke/repos-pvc.yml -n polyaxon
```

## Initialize Helm and grant RBAC
kubectl --namespace kube-system create sa tiller
kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
helm init --service-account tiller

## Deploy Polyaxon
```
helm repo add polyaxon https://charts.polyaxon.com
helm repo update
helm install polyaxon/polyaxon --name=polyaxon --namespace=polyaxon -f gke/polyaxon-config.yml

export POLYAXON_IP=$(kubectl get svc --namespace polyaxon polyaxon-polyaxon-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export POLYAXON_HTTP_PORT=80
export POLYAXON_WS_PORT=80

echo http://$POLYAXON_IP:$POLYAXON_HTTP_PORT
polyaxon config set --host=$POLYAXON_IP --http_port=$POLYAXON_HTTP_PORT  --ws_port=$POLYAXON_WS_PORT
```

## Create project and launch experiment
```
polyaxon login --username=root --password=rootpassword


polyaxon project create --name=finefood --description='Polyaxon Amazon Fine Food.'

polyaxon init finefood
polyaxon dashboard

polyaxon run -u -f polyaxonfile.yml

```


## Check logs

```
gcloud --project "amazonfinefood" compute ssh --ssh-flag=-L3000:localhost:3000 --zone=us-central1-b polyaxon-nfs-vm
```

```
cd /data/logs/root/finefood/independents
```

