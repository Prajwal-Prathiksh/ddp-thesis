# k-ϵ Model

  

## SOC Governing Equations

  

$$ P = \frac{\rho_o c_o^2}{\gamma} \bigg( \bigg( \frac{\rho}{\rho_o} \bigg)^\gamma - 1 \bigg) $$

  

$$ \frac{D\rho}{Dt} = -\rho \nabla . \vec{u} $$

  

$$ \frac{D\vec{u}}{Dt} = - \frac{\nabla P}{\rho} + \nu \nabla^2 \vec{u} $$

  
  

## RANS Governing Equations

  

$$ \boxed{P = \frac{\rho_o c_o^2}{\gamma} \bigg( \bigg( \frac{\rho}{\rho_o} \bigg)^\gamma - 1 \bigg)}$$

  

$$ \boxed{\frac{D\rho}{Dt} = -\rho \nabla . \vec{u}}$$

  

$$ \boxed{\frac{D\vec{u}}{Dt} = - \frac{\nabla P}{\rho} + \nu \nabla^2 \vec{u} + \nabla . \underline{\tau} \rho^{-1} }$$

  

$$ \boxed{\underline{\tau}\rho^{-1} = 2C_d \frac{k^2}{\epsilon} \underline{S} - \frac{2}{3}k\underline{I}} = 2\nu_t\underline{S} - \frac{2}{3}k\underline{I} $$

  

$$ \boxed{\nu_t = C_d \frac{k^2}{\epsilon}} $$

  

$$ \underline{S} = \frac{1}{2} \bigg( \nabla \vec{u} + (\nabla \vec{u})^T \bigg) $$

  

$$ \boxed{P_k = C_d \sqrt{2<\underline{S}, \underline{S}>}} $$

  

$$ \frac{Dk}{Dt} = \nabla . \bigg( \frac{\nu_t}{\sigma_k} \nabla k \bigg) + P_k - \epsilon $$

$$ \implies \frac{Dk}{Dt} = \frac{C_d}{\sigma_k} \nabla . \bigg ( \frac{k^2}{\epsilon} \nabla k \bigg) + P_k - \epsilon $$

$$ = \frac{C_d}{\sigma_k} \bigg( \frac{k^2}{\epsilon} \nabla^2 k + \nabla k . \nabla \bigg( \frac{k^2}{\epsilon} \bigg) \bigg) + P_k - \epsilon $$

$$ \boxed{ \frac{Dk}{Dt} = \frac{C_d}{\sigma_k} \bigg( \frac{k^2}{\epsilon} \nabla^2 k + \frac{2k}{\epsilon} (\nabla k)^2 - \frac{k^2}{\epsilon^2} \nabla k . \nabla \epsilon \bigg) + P_k - \epsilon }$$

  

$$ \frac{D\epsilon}{Dt} = \nabla . \bigg( \frac{\nu_t}{\sigma_\epsilon} \nabla \epsilon \bigg) + C_{1\epsilon} \frac{\epsilon}{k} P_k - C_{2\epsilon} \frac{\epsilon^2}{k} $$

$$ \implies \frac{D\epsilon}{Dt} = \frac{C_d}{\sigma_\epsilon} \nabla . \bigg( \frac{k^2}{\epsilon} \nabla \epsilon \bigg) + C_{1\epsilon} \frac{\epsilon}{k} P_k - C_{2\epsilon} \frac{\epsilon^2}{k} $$

$$ = \frac{C_d}{\sigma_\epsilon} \bigg( \frac{k^2}{\epsilon} \nabla^2 \epsilon + \nabla \epsilon . \nabla \bigg( \frac{k^2}{\epsilon} \bigg) \bigg) + C_{1\epsilon} \frac{\epsilon}{k} P_k - C_{2\epsilon} \frac{\epsilon^2}{k} $$

$$ \boxed{ \frac{D\epsilon}{Dt} = \frac{C_d}{\sigma_\epsilon} \bigg( \frac{k^2}{\epsilon} \nabla^2 \epsilon + \frac{2k}{\epsilon} \nabla k \nabla \epsilon - \frac{k^2}{\epsilon^2} (\nabla \epsilon)^2 \bigg) + C_{1\epsilon} \frac{\epsilon}{k} P_k - C_{2\epsilon} \frac{\epsilon^2}{k}} $$

  

$$ \boxed{(C_d, \sigma_k, \sigma_\epsilon, C_{1\epsilon}, C_{2\epsilon}) = (0.09, 1.0, 1.3, 1.44, 1.92)} $$

### Initialisation
Initial values for $k-\epsilon$ 
	- Ref: https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence-ras-k-epsilon.html#sec-turbulence-ras-k-epsilon-initialisation
$$ k_{init} = \frac{3}{2}(\underline{I}|u_{ref}|^2) $$
where $\underline{I}$ is the turbulence intensity (%), and $u_{ref}$ is the reference flow speed.
$$ \epsilon_{init} = \frac{C_{\mu}^{0.75} k^{1.5}}{L} $$
where, $C_{\mu}$ is a model constant $=0.09$, and $L$ is the length scale. 