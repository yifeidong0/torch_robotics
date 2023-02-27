import torch
import time

from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda, DifferentiableAllegroHand, DifferentiableShadowHand, DifferentiableTiagoDualHoloMove, DifferentiableUR10


if __name__ == "__main__":

    batch_size = 10
    device = "cpu"
    print("===========================Panda Model===============================")
    panda_kin = DifferentiableFrankaPanda(device=device)
    panda_kin.print_link_names()
    print(panda_kin.get_joint_limits())
    print(panda_kin._n_dofs)
    time_start = time.time()
    q = torch.rand(batch_size, panda_kin._n_dofs).to(device)
    q.requires_grad_(True)
    data = panda_kin.compute_forward_kinematics_all_links(q)
    print(data.shape)
    time_end = time.time()
    print("Computational Time {}".format(time_end - time_start))

    print("===========================UR10 Model===============================")
    ur10_kin = DifferentiableUR10(device=device)
    ur10_kin.print_link_names()
    print(ur10_kin.get_joint_limits())
    print(ur10_kin._n_dofs)
    time_start = time.time()
    q = torch.rand(batch_size, ur10_kin._n_dofs).to(device)
    q.requires_grad_(True)
    data = ur10_kin.compute_forward_kinematics_all_links(q)
    print(data.shape)
    time_end = time.time()
    print("Computational Time {}".format(time_end - time_start))

    print("===========================Tiago Model===============================")
    tiago_kin = DifferentiableTiagoDualHoloMove(device=device)
    tiago_kin.print_link_names()
    print(tiago_kin.get_joint_limits())
    print(tiago_kin._n_dofs)
    time_start = time.time()
    q = torch.rand(batch_size, tiago_kin._n_dofs).to(device)
    q.requires_grad_(True)
    data = tiago_kin.compute_forward_kinematics_all_links(q)
    print(data.shape)
    time_end = time.time()
    print("Computational Time {}".format(time_end - time_start))

    print("===========================Shadow Hand Model===============================")
    hand = DifferentiableShadowHand(device=device)
    hand.print_link_names()
    print(hand.get_joint_limits())
    print(hand._n_dofs)
    time_start = time.time()
    q = torch.rand(batch_size, hand._n_dofs).to(device)
    q.requires_grad_(True)
    data = hand.compute_forward_kinematics_all_links(q)
    print(data.shape)
    time_end = time.time()
    print("Computational Time {}".format(time_end - time_start))

    print("===========================Allegro Hand Model===============================")
    hand = DifferentiableAllegroHand(device=device)
    hand.print_link_names()
    print(hand.get_joint_limits())
    print(hand._n_dofs)
    time_start = time.time()
    q = torch.rand(batch_size, hand._n_dofs).to(device)
    q.requires_grad_(True)
    data = hand.compute_forward_kinematics_all_links(q)
    print(data.shape)
    time_end = time.time()
    print("Computational Time {}".format(time_end - time_start))
