name: Build Docker image
run: |
docker build -t my-dockerhub-youssefhemdane/my-app:${{ github.sha }} .
docker tag my-dockerhub-youssefhemdane/my-app:${{ github.sha }} my-dockerhub-youssefhemdane/my-
app:latest

- name: Push Docker image
run: |
docker push my-dockerhub-youssefhemdane/my-app:${{ github.sha }}
docker push my-dockerhub-youssefhemdane/my-app:latest
