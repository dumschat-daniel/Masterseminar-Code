This Repository contains the Code Snippets for testing and evaluating different parts and mechanisms of quantization for the module "Masterseminar"

channel_wise_quantization: This code snippet compares channel wise and global quantization and shows the advantages that channel wise quantization has and what issues it avoids.

model_size_fp32: a small helper showing the native model sizes of the used backbones in fp32.

precision_comparison_demo: Compares a Simple MLP in 32f and 16f

quantization_demo: A small demo demonstrating the idea of quantization on a few weights

quantization_methods: Comparison of three typical quantization methods: min-max, percentile clipping and entropy based.

weights_vs_task_optimization: This code snippet shows the idea of the authors and why optimizing in parameter space is not necessarly the best way to optimize the whole task loss.

setup_resnet18_for_imagenette: This shows the code to produce a Resnet18 base that can be used for Brecq experiments with the imagenette datast.
