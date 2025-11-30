Run:
```
python main.py --config configs/default.yaml
```
or:
```cmd
python main.py --config configs/default.yaml \
               --model_type lstm \
               --batch_size 32 \
               --learning_rate 0.001 \
               --hidden_size 64
```
To run train + rollout and plot:

```
python main.py --config configs/default.yaml --plot
```