# Development

## Testing your Dockerfile locally

Navigate to the directory containing this file and the `.dockerfile`. Likely:
```bash
cd tutorials/tutorial-01-basics/
```

Then build and run the docker image:
```bash
docker build -t marimo-app .
docker run -it --rm -p 7860:7860 marimo-app
```
YOu need to install and configure docker on your machine. See [here](https://docs.docker.com/get-docker/) for more information.