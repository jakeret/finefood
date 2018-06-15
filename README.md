# finefood
Amazon Fine Food and Polyaxon


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

