注意：pipeline中用到GroundingDINO fp16模型，用sail推理会引起芯片Fault。必须用tpu_perf。

把tpu_perf包的infer.py替换为`./replace-this-file/infer.py`。
```bash
cp ./replace-this-file/infer.py /home/linaro/.local/lib/python3.8/site-packages/tpu_perf/ # (若不是该路径可`python -c"import tpu_perf.infer as infer; print(infer.__file__)"`看下
```