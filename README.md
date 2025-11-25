# Robo-ly

TODO:
- [ ] Be able to run Eureka
- [ ] Create new tasks

```
cd ~/Eureka/eureka/utils
python prune_env.py cartpole_spin

cd ~/Eureka/eureka
python eureka.py env=cartpole_spin

python eureka.py env=cartpole_spin checkpoint=/home/ttr/Eureka/eureka/outputs/eureka/2025-10-08_21-17-18/policy-2025-10-08_21-18-32/runs/CartpoleSpinGPT-2025-10-08_21-18-32/nn/CartpoleSpinGPT.pth env.description="spin as fast as you can in one direction"
```