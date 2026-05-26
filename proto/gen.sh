protoc -I proto \
  --go_out=be \
  --go_opt=module=github.com/YoDobchev/pravopisly/be \
  --go-grpc_out=be \
  --go-grpc_opt=module=github.com/YoDobchev/pravopisly/be \
  proto/pravopisly.proto

python -m grpc_tools.protoc \
  -I proto \
  --python_out=model/pb \
  --grpc_python_out=model/pb \
  proto/pravopisly.proto