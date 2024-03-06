## How to use?

1. Follow the instruction in [InstantAvatar](https://github.com/tijiang13/InstantAvatar) to run the code in` custom_capture`.    The visual code is not the same but better. We found it usually need to change some params to fit the case. If the results are not satisfactory, change the params in `refine-smpl.py` to get a better one.

2. Use `filetransfer.sh` to convert the result so the main code could train it.

3. If you want to use the pose in the video to drive someone else, use `filetransfer-386shape.sh`.
4. Do not forgrt to change the path. Both in the .sh file and `maincode/core/data/dataset_args.py`.





## Citation	

```
@misc{ma2023humannerfse,
      title={HumanNeRF-SE: A Simple yet Effective Approach to Animate HumanNeRF with Diverse Poses}, 
      author={Caoyuan Ma and Yu-Lun Liu and Zhixiang Wang and Wu Liu and Xinchen Liu and Zheng Wang},
      year={2023},
      eprint={2312.02232},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```