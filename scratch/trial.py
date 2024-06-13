from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as ptv
from torch.nn.utils import vector_to_parameters as vtp
from torch import linalg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fun = nn.CrossEntropyLoss()

# loss_fun = SparsemaxLoss()


def hermin(xl, fl, fpl, xu, fu, fpu):
    dx = xu - xl  # use f = fl + fpl*dt + c*dt^2+d*dt^3 with dt = (x-xl)/(xu-xl)
    df = fu - fl
    dfp = fpu - fpl
    afp = (fpu + fpl) / 2.
    d = 6. * (afp - df / dx)  # cubic d should be non negative
    c = dfp - d  # quadratic term should mostly be positive
    c0 = c
    d0 = d
    c /= - 2. * fpl
    d /= -fpl  # allows for d = 0, i.e. exact quadratic interpolation
    discriminant = c * c + d
    if discriminant < 0.0:
        print(
            '\n trouble in Hermitian interpolation, c {:.2E}, d {:.2E}, discriminant {:.2E}'.format(c, d, discriminant))
        print(
            'The variables are: xl {:.2E}, fl {:.2E}, fpl {:.2E}, xu {:.2E}, fu {:.2E}, fpu {:.2E}'.format(xl, fl, fpl,
                                                                                                           xu, fu, fpu))
        return xu, 0.
    rho = 1.0 / (c + math.sqrt(discriminant))  # but fpl must be positive, which is part of lower condition
    dx *= rho
    del_f = df / dx

    return xl + dx, del_f


def vectorize_grad(mdl):
    views = []
    for p in mdl.parameters():
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().detach().view(-1)
        else:
            view = p.grad.detach().view(-1)
        views.append(view)
    return torch.cat(views, 0)


def evaluate(model_running, data_running, target_running, eta_running=0., dist=None, loss=loss_fun, regc = None):
    l2_reg = torch.tensor(0., requires_grad=True)
    tmp = None
    if eta_running != 0. and dist is not None:
        tmp = ptv(model_running.parameters())
        upd = ptv(model_running.parameters()) - torch.mul(dist, eta_running)
        vtp(upd, model_running.parameters())
    for params in model_running.parameters():
        params.grad = None
    err = loss(model_running(data_running), target_running)
    if regc is not None:
        for name, param in model_running.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 1)
        err = err + regc * l2_reg
    err.backward()
    gradient = vectorize_grad(model_running)
    if eta_running != 0.:
        vtp(tmp, model_running.parameters())
    return err.item(), gradient


def update(model_running, eta_running, dist=None):
    if eta_running == 0. or dist is None:
        return
    else:
        upd = ptv(model_running.parameters()) - torch.mul(dist, eta_running)
        vtp(upd, model_running.parameters())
    return


def get_data(batch_size_train, batch_size_test):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])

    trainset = datasets.CIFAR10(root='../../../data', train=True,
                                download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='../../../data', train=False,
                               download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size_test, num_workers=8, pin_memory=True)
    return train_loader, test_loader


def train(epochs, train_loader, test_loader, max_line_search=5):
    torch.cuda.empty_cache()
    loss_val = []
    loss_train = []
    avg_loss_train = []
    accur = []
    accur_train = []
    etas = []
    etab = []
    etat = []
    dgrads = []
    g_norm = []
    d_norm = []
    phi_avgs = []
    phi_primes = []
    g_norm_avgs = []
    lines = []
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
    # model = ResNet18()
    # model = models.vgg16()
    model.to(device)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    delta_tol = 1.0e-4
    eta_tol = 1.0e-6
    mach_eps = 1.0e-14
    d = None
    m1 = 0.3
    m2 = 0.3
    m3 = 0.95
    phi_avg = 0.
    phi_prime_avg = 0.
    g_norm_avg = 0.
    counter = 1
    m_counter = 1

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Initial evaluation
            output = model(data)
            phi_b, g = evaluate(model, data, target)
            g_norm_b = torch.norm(g).item()
            g_norm.append(g_norm_b)
            # total_train_loss += phi_b * len(data)
            loss_train.append(phi_b)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accur_train.append(correct / len(data))
            print(f" The batch {batch_idx} loss is {phi_b} and accuracy is {correct}.")

            if d is None:
                d = g

            if g_norm_b == 0.:
                break

            eta_b = eta_s = 0.
            eta = 1.
            eta_u = math.inf
            eta_p = math.inf
            switch = False
            sufficient_decrease = False
            sufficient_decrease_b = False
            active_kink = False
            g_l = None

            it_inner = 0
            phi_u = math.inf
            phi_b_grad = - torch.dot(d, g).item()

            phi_u_grad = math.inf

            if phi_b_grad > 0.:
                d = - d
                phi_b_grad = - phi_b_grad
                switch = True

            phi_start = phi_b
            dgrads.append(phi_b_grad)
            phi_avg += (phi_b - phi_avg) / (float(m_counter))
            phi_prime_avg += (phi_b_grad - phi_prime_avg) / (float(m_counter))
            g_norm_avg += (g_norm_b - g_norm_avg) / (float(m_counter))
            # phi_avg = wiener(np.asarray(loss_train), counter + 1)[-1]
            # phi_prime_avg = wiener(np.asarray(dgrads), counter + 1)[-1]
            # g_norm_avg = wiener(np.asarray(g_norm), counter + 1)[-1]
            counter += 1
            m_counter += 1
            phi_avgs.append(phi_avg)
            phi_primes.append(phi_prime_avg)
            g_norm_avgs.append(g_norm_avg)

            alphas = []
            betas = []
            ts = []
            ps = []
            pas = []
            pps = []
            ppas = []
            A = []
            B = []
            S = []
            I = []
            E = []
            N = []

            while eta_u - eta_b > eta_tol and it_inner < max_line_search:
                it_inner += 1
                phi, phi_g = evaluate(model, data, target, eta, d)
                phi_grad = torch.dot(d, phi_g).item()
                gn = torch.norm(phi_g).item()
                ps.append(phi)
                pps.append(phi_grad)
                pas.append(phi_avg)
                ppas.append(phi_prime_avg)
                alphas.append(eta_b)
                betas.append(eta_u)
                ts.append(eta)
                N.append(gn)

                if switch:
                    phi_grad = - phi_grad
                    S.append(1)
                else:
                    S.append(0)

                if phi > phi_b + phi_b_grad:
                    eta_u = eta
                    phi_u = phi
                    phi_u_grad = phi_grad
                    eta *= 0.5
                    A.append(0)
                    B.append(0)

                else:
                    sufficient_decrease = True
                    eta_s = eta
                    g = phi_g.clone()
                    A.append(1)
                    B.append(1)
                    I.append(0)
                    E.append(0)
                    break

            lines.append([alphas, betas, ts, ps, pas, pps, ppas, A, B, S, I, E, N])

            etat.append(eta_u)
            etab.append(eta_b)

            if sufficient_decrease:
                etas.append(eta_s)
            else:
                eta_s = ts[np.argmin(ps)]
                etas.append(eta_s)

            update(model, eta_s, d)
            if switch:
                d = -d

            d = (pow(linalg.norm(g), 2) * d + pow(linalg.norm(d), 2) * g) / (
                    pow(linalg.norm(g), 2) + pow(linalg.norm(d), 2))

            dnorm = linalg.norm(d).cpu().detach().numpy()
            d_norm.append(dnorm)

            m1 = max(m1, 1. - dnorm)

            # emp, g = evaluate(model, data, target, eta, d)
            # print("\t dir ders initial and final ", phi_prime_start, -torch.dot(d, g).item())

            # phi_t_grad_l = torch.dot(d, g_u).item()
            #
            # else:
            #   etas.append(0.)

            if linalg.norm(d) < mach_eps:
                print("Direction too short. Exiting with norm(d): ", linalg.norm(d))
                print("gradient norm is: ", linalg.norm(g))
                return loss_val, loss_train, avg_loss_train, accur, accur_train, etas, etab, etat, g_norm, d_norm, dgrads, phi_avgs, phi_primes, g_norm_avgs, lines

        model.eval()
        test_loss = 0.
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.float(), target.to(dtype=torch.long)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fun(output, target).item() * len(data)  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        loss_val.append(test_loss)
        test_acc = correct / len(test_loader.dataset)
        accur.append(test_acc)

        print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        if test_acc > 0.99:
            print("\n Accuracy: ", test_acc, ". Threshold crossed. Exiting\n")
            break

    torch.save(model.state_dict(), './state_dict.pt')

    return loss_val, loss_train, avg_loss_train, accur, accur_train, etas, etab, etat, g_norm, d_norm, dgrads, phi_avgs, phi_primes, g_norm_avgs, lines


def main():
    bs = 128
    bst = 100
    t_l, tst_l = get_data(bs, bst)
    eps = 100
    # sys.stdout = open("./output.txt", 'wt')

    l_v, l_t, al_t, acc, acct, etas, etab, etat, \
    g_norm, d_norm, dgrads, ps, pps, gnas, lines = train(eps, t_l, tst_l, 5)

    np.save("./validation_losses.txt", np.asarray(l_v))
    np.save("./training_losses.txt", np.asarray(l_t))
    np.save("./average_training_losses.txt", np.asarray(al_t))
    np.save("./accuracy.txt", np.asarray(acc))
    np.save("./accuracy_training.txt", np.asarray(acct))
    np.savetxt('./etas.txt', np.asarray(etas))
    np.savetxt('./etab.txt', np.asarray(etab))
    np.savetxt("./etat.txt", np.asarray(etat))
    np.savetxt("./gnorm.txt", np.asarray(g_norm))
    np.savetxt("./dnorm.txt", np.asarray(d_norm))
    np.savetxt("./dgrads.txt", np.asarray(dgrads))
    np.savetxt("./phis.txt", np.asarray(ps))
    np.savetxt("./phips.txt", np.asarray(pps))
    np.savetxt("./gnas.txt", np.asarray(gnas))


if __name__ == '__main__':
    main()
