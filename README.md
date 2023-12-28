# Trans-Net: A transferable pretrained neural networks based on temporal domain decomposition for solving partial differential equations

Physics-Informed Neural Networks (PINNs) have provided a novel direction for solving partial differential equations (PDEs) and have achieved significant advancements in the field of scientific computing. PINNs effectively incorporate the physical constraints of equations into the loss function, enabling neural networks to learn and approximate the behavior of physical systems by optimizing the loss function. According to the existing research, we propose a transferable pretrained neural networks framework based on temporal domain decomposition to solve partial differential equations. Specifically, we divide the domain into multiple subdomains according to time, each of which is solved using an individual neural network. Subsequently, the trained neural networks and the predicted values at the common boundary are used as the pretrained model and initial values for the next subdomain, respectively. This not only improves the subnetwork's prediction accuracy and convergence rate but also reduces network parameters required by the subnetwork. Finally, we present a series of classical numerical experiments including one-dimensional, two-dimensional, and three-dimensional partial differential equations. The experimental results indicate that the proposed method outperforms existing approaches in terms of accuracy and efficiency.

The environment required for pytorch and tensenflow is located in pytorch.txt and tensenflow.txt, respectively

It is worth noting that we used the source code provided by the PINNs authors to solve the one-dimensional Burgers equation, and obtained a relative L2 error of 4.49 e-03 in a configuration of 5 hidden layers and 20 neurons, which is slightly different from the 1.3 e-03 in the PINNs paper.
