from __future__ import print_function
from fenics import *
import os
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import time
from math import *
import sympy as sym

from numpy.random import rand

from operator import mul
from functools import reduce

from scipy.integrate import odeint

from reaction_diffusion_source_functions import *

def prod(seq):
    return reduce(mul, seq) if seq else 1

def convert_list_to_tuples(parameterList):
    
    name_key = 0
    value_key = 1
    values =[]
    names =[]
    
    for parameter in parameterList:
        names.append(parameter[name_key])
        values.append(parameter[value_key])
    
    return tuple(names),tuple(values)

def form_reaction_term(reactions, names,fenicsNames='empty'):
    # create symbols for reactants
    if fenicsNames == 'empty':
        symbs = sym.symbols(names, real=True, nonnegative=True)
    
    else:
        symbs = sym.symbols(fenicsNames, real=True, nonnegative=True)

    # map between reactant symbols and keys in r_stoich, net_stoich
    c = dict(zip(names, symbs))
    f = {n: 0 for n in names}
    k = []
    for coeff,constValue,kineticType, r_stoich, net_stoich in reactions:
        k.append(sym.S(coeff))
        
        r=0
        if(kineticType=='Mass_Action'): 
            r = k[-1]*prod([c[rk]**p for rk, p in r_stoich.items()])
        
        for net_key, net_mult in net_stoich.items():
            f[net_key] += net_mult*r
    
    return [f[n] for n in names], symbs, tuple(k)


def determine_fenics_names(names):

    # convert the names into U_N[i] version
    fenicsNames = []
    for index in range(0,len(names)):
        fenicsNames.append('U_N['+str(index)+']')

    return fenicsNames


def convert_source_term(names,source_term):
    # convert the names into U_N[i] version, search through symbolicExpression 
    # replace names with converted form

    fenicsNames = determine_fenics_names(names)
    symbs = sym.symbols(fenicsNames, real=True, nonnegative=True)
    convertedExpressionArray =[]

    for stringIndex in range(0,len(source_term)):
        convertedExpression = sym.sympify(source_term[stringIndex])
         
        for nameIndex in range(0,len(names)):
            convertedExpression = convertedExpression.subs(names[nameIndex],sym.Symbol(fenicsNames[nameIndex]))

        convertedExpressionArray.append(convertedExpression)

    return convertedExpressionArray


def return_rate_constants(reactions,kVec):
    
    ks=[]
    rateNameKey=0
    rateKey = 1
    
    for k in kVec:
        ks.append(0)
        
    for reaction in reactions:
        i=0
        for k in kVec:
            if(sym.symbols(reaction[rateNameKey])==k):
                ks[i]=reaction[rateKey]
                break
            i=i+1
    
    return ks

def coefficient_substitute(reactionString, reactions):
    rateNameKey=0
    rateKey = 1

    for reaction in reactions:
        reactionString=reactionString.replace(reaction[rateNameKey], str(reaction[rateKey]))
    
    return reactionString


def general_coefficient_substitute(pdeString, coefficients):
    rateNameKey=0
    rateKey = 1

    for coeff in coefficients:
        pdeString=pdeString.replace(coeff[rateNameKey], str(coeff[rateKey]))
    
    return pdeString



def make_output_directory(file_root, simulation_tag=0):
    path = file_root+simulation_tag
    if not os.path.isdir(path):
        os.makedirs(path) 
    return path

# test whether a point x is on a boundary
tol = 1E-14

def boundary_DC(x, on_boundary):
    return on_boundary

# alternatively
# return near(x[0], 0, tol) or near(x[1], 0, tol) \
# or near(x[0], 1, tol) or near(x[1], 1, tol)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0, tol)
    
    
class BoundaryX0(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)
    
class BoundaryX1(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1, tol)
    
class BoundaryY0(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    
class BoundaryY1(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1, tol)


#define intial condition
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        np.random.seed()
        self.IC = kwargs["IC"]
        self.Randomise = kwargs["Randomise"]
        self.DomainType = kwargs["DomainType"]
        self.numberOfSpecies = len(kwargs["IC"])
        super().__init__()
            
    def eval(self, values, x):

        if self.DomainType == "delta_peak":
            if x[0] <= 0.1:
                for index in range(0,self.numberOfSpecies):
                    values[index] = self.IC[index]
            else:
                for index in range(0,self.numberOfSpecies):
                    values[index] = 0.0

        elif self.DomainType == "centre_peak":
            if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
                for index in range(0,self.numberOfSpecies):
                    values[index] = self.IC[index]
            else:
                for index in range(0,self.numberOfSpecies):
                    values[index] = 0.0

        else:
            if self.Randomise==1:
                for index in range(0,self.numberOfSpecies):
                    values[index] = self.IC[index] + (np.random.random() - 0.5)

            else:
                for index in range(0,self.numberOfSpecies):
                    values[index] = self.IC[index]


    def value_shape(self):
        return (self.numberOfSpecies,)



    
def setup_function_space(mesh,variable_names,simulation_settings,state_variables):

    [simulationType,domainType,output_dir_path] = simulation_settings
    [variable_names,var_init_conds] = state_variables
    
    #P1=FiniteElement('CG',triangle,1)#or 3)
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)

    # Define function space for system of concentrations
    element = MixedElement([P1 for i in variable_names])

    # Function space to approximate solution in 
    V = FunctionSpace(mesh, element)
    U = Function(V) # function to hold the next variable values

    return U,V    


def determine_names_from_reactions(reactions):
    
    names=[]
    fenicsNames=[]
    rhsStoichKey = 3
    netStoichKey = 4

    for reaction in reactions:
        for name in reaction[rhsStoichKey]:
            if name not in names:
                names.append(name)

        for name in reaction[netStoichKey]:
            if name not in names:
                names.append(name)
    
    for index in range(0,len(names)):
        fenicsNames.append('U_N['+str(index)+']')
        #fenicsNames.append('u'+str(index))
    
    return names,fenicsNames


def reactionDiffusionPDE(U,U_N,V,mesh,dt,dx,diffusionValues,reactions):
    
    (x, y) = SpatialCoordinate(mesh)
    dt_inv = 1 / dt

    # define test functions
    vTuple = TestFunctions(V)
    #print("here")
    names, fenicsNames = determine_names_from_reactions(reactions)
    #print(names)
    #print(fenicsNames)
    ydot, y, k = form_reaction_term(reactions, names,fenicsNames)
    #print(ydot)

    ccodeTuple = []
    for index in range(0,len(names)):
        f_code = sym.printing.ccode(ydot[index])
        #print(ydot[index])
        f_code = coefficient_substitute(f_code,reactions)
        #print(f_code)
        ccodeTuple.append(Expression(f_code, degree=2, U_N=U_N))

        
    F=0
    for index in range(0,len(names)):

        F += dt_inv*(U[index] - U_N[index])*vTuple[index]*dx
        F += Constant(diffusionValues[index])*inner(grad(U[index]), grad(vTuple[index]))*dx

        F -= ccodeTuple[index]*vTuple[index]*dx

    return F


def FisherKPPPDE(U,U_N,V,mesh,dt,dx,diffusionValues):
    
    r = Constant(1.0)
    
    (x, y) = SpatialCoordinate(mesh)
    dt_inv = 1 / dt

    # define test functions
    vTuple = TestFunctions(V)

        
    F=0
    for index in range(0,len(diffusionValues)):

        F += dt_inv*(U[index] - U_N[index])*vTuple[index]*dx
        F += Constant(diffusionValues[index])*inner(grad(U[index]), grad(vTuple[index]))*dx

        F -= U_N[index]*(1-U_N[index])*vTuple[index]*dx

    return F



def pde_weak_formulation(U,U_N,V,mesh,state_variables,simulation_settings,simulation_parameters,measures,diffusionValues,reactions):
    (x, y) = SpatialCoordinate(mesh)
    
    [simulationType,domainType,output_dir_path] = simulation_settings
    [t_end,dt] = simulation_parameters
    [variable_names,var_init_conds] = state_variables

    F=0
    

    F=reactionDiffusionPDE(U,U_N,V,mesh,simulation_parameters[2],measures[0],diffusionValues,reactions)

    return F

def general_weak_formulation(U,U_N,V,mesh,state_variables,simulation_settings,simulation_parameters,measures,diffusionValues,source_term,parameters):
   
    (x, y) = SpatialCoordinate(mesh)
    
    [simulationType,domainType,output_dir_path] = simulation_settings
    [t_end,dt] = simulation_parameters
    [variable_names,var_init_conds] = state_variables

    F=0
    
   
    if(simulationType=='General'):
       
        F=PDE_from_symbolic(U,U_N,V,mesh,simulation_parameters[1],measures[0],diffusionValues,variable_names,source_term,parameters)

    elif(simulationType=='Reaction'):

        F=reactionDiffusionPDE(U,U_N,V,mesh,simulation_parameters[1],measures[0],diffusionValues,source_term)

    else:
        print("Pde type unknown")
    
    return F

def PDE_from_symbolic(U,U_N,V,mesh,dt,dx,diffusionValues,names,source_term,parameters):
    
   
    (x, y) = SpatialCoordinate(mesh)
    dt_inv = 1 / dt
    
    # define test functions
    vTuple = TestFunctions(V)

    #names, fenicsNames = determine_names_from_reactions(reactions)

    # need a version of "fenicsNames" to substitute the variables
    symbolicExpression = convert_source_term(names,source_term)

    #ydot, y, k = form_reaction_term(reactions, names,fenicsNames)
    #r,c,a = sym.symbols('r, c, a', negative=False)

    ccodeTuple = []
    for index in range(0,len(symbolicExpression)):
        f_code = sym.printing.ccode(symbolicExpression[index])
        f_code = general_coefficient_substitute(f_code,parameters)
        ccodeTuple.append(Expression(f_code, degree=len(symbolicExpression), U_N=U_N))

  
    F=0
    for index in range(0,len(symbolicExpression)):

        F += dt_inv*(U[index] - U_N[index])*vTuple[index]*dx
        F += Constant(diffusionValues[index])*inner(grad(U[index]), grad(vTuple[index]))*dx

        F -= ccodeTuple[index]*vTuple[index]*dx


    return F




def pde_weak_formulation_wave(U,U_N,V,mesh,state_variables,simulation_settings,simulation_parameters,measures,diffusionValues):
    (x, y) = SpatialCoordinate(mesh)
    
    [simulationType,domainType,output_dir_path] = simulation_settings
    [t_end,dt] = simulation_parameters
    [variable_names,var_init_conds] = state_variables

    F=0
    

    F=FisherKPPPDE(U,U_N,V,mesh,simulation_parameters[2],measures[0],diffusionValues)

    return F


def retrieve_rate_constants(reactions):
    
    k_name_key=0
    k_value_key=1
    
    k_vals=[]
    k_val_names=[]
    
    for reaction in reactions:
        k_val_names.append(reaction[k_name_key])
        k_vals.append(reaction[k_value_key])
        
    return k_val_names,k_vals



def General_pde_solver(source_term,parameterList,variable_names,diffusionValues,var_init_conds,boundary_conditions,file_root,simulation_tag,simulationType,domainType,times,meshDims):

    output_dir_path = make_output_directory(file_root+simulationType+"/",simulation_tag)

    try:

        [nx,ny]=meshDims
        [t_initial,t_end,dt]=times
        simulation_settings = [simulationType,domainType,output_dir_path]
        simulation_parameters = [t_end,dt]
        state_variables = [variable_names,var_init_conds]
        #boundary_conditions = []
        mesh_settings = [[nx,ny]]
      
        # form the mesh covering the computational domain
        mesh = UnitSquareMesh(nx, ny)
   
        # boundary conditions
        boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        boundary_markers.set_all(9999)
    
        bx0 = BoundaryX0()
        bx0.mark(boundary_markers, 0)
        bx1 = BoundaryX1()
        bx1.mark(boundary_markers, 1)
        by0 = BoundaryY0()
        by0.mark(boundary_markers, 2)
        by1 = BoundaryY1()
        by1.mark(boundary_markers, 3)

        # Redefine boundary integration measure
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
            
        # Collect Dirichlet conditions
        BCs = []
        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                bc = DirichletBC(V, [Constant(j) for j in boundary_conditions[i]['Dirichlet']],
                                boundary_markers, i)
                bcs.append(bc)
                
  
        # set up the test nad trial function spaces
        U,V = setup_function_space(mesh,variable_names,simulation_settings,state_variables)

        # set initial conditions
        #U,u_n = setup_initial_conditions(U,V,state_variables,simulation_settings)
      
        u_n = Function(V)
        u_n.interpolate(InitialConditions(IC = state_variables[1],Randomise = 1, DomainType = domainType))
        U.assign(u_n)
      
        # define measure across domain
        dx = Measure('dx', domain=mesh)

        measures = [dx,ds]

        F = 0

        # define test functions
        vTuple = TestFunctions(V)

    
        # Collect Neumann integrals
        for i in boundary_conditions:
            if 'Neumann' in boundary_conditions[i]:
                index=0
                for value in boundary_conditions[i]['Neumann']:
                    index +=1
                    if value != 0:
                        F += value*vTuple[index]*ds(i)
                

        # Simpler Robin integrals
        for i in boundary_conditions:
            if 'Robin' in boundary_conditions[i]:
                index=0
                for value in boundary_conditions[i]['Robin']:
                    index +=1
                    [r, s] = value
                    F += r*(u - s)*vTuple[index]*ds(i)


        # add the weak formulations of the separate PDEs
        F += general_weak_formulation(U,u_n,V,mesh,state_variables,simulation_settings,simulation_parameters,measures,diffusionValues,source_term,parameterList)
   

        # add Newmann and Robin BC's, define dirichlet BC
        # boundary conditions
        #[F, boundary_conditions] = setup_boundary_conditions(F,U,U_N,V,boundary_markers,measures,state_variables,simulation_settings,boundary_conditions_vec)


        # Assemble the (non-)linear system 
        FA = lhs(F)
        Fb = rhs(F)
        A = assemble(FA, keep_diagonal = True)

        # Create VTK files for visualization output
        vtk_files = []
        for var_name in variable_names:
            vtk_files.append(File(str(output_dir_path+'/'+var_name+'_.pvd')))

            

        # run through the simulation step by step
        # Solving the problem in time
        t = t_initial
        plotSampleRate = 10
        n=0
        plotIndex=0
        while t< t_end:
            # Update time-step
            t += dt

            # Solve linear variational problem for time step
            b = assemble(Fb)
            solve(F==0,  U, BCs)  

            # Save solution to file (VTK)
            for var_name in variable_names:
                # specific variable solution specfied by the index
                index = variable_names.index(var_name)
                vtk_files[index] << (U.sub(index), t)

            # update the current variable vector for the next simulation
            u_n.assign(U)
            if(n==plotIndex*plotSampleRate):
                plot(U)
                plt.show()
                plotIndex=plotIndex+1
            n=n+1


    except:
        print("General_pde_solver(): Fenics solver error")

    return output_dir_path


def Reaction_pde_solver(reactions,variable_names,diffusionValues,var_init_conds,boundary_conditions,file_root,simulation_tag,simulationType,domainType,times,meshDims):


    output_dir_path = make_output_directory(file_root+simulationType+"/",simulation_tag)

    try:
  

        [nx,ny]=meshDims
        [t_initial,t_end,dt]=times
        simulation_settings = [simulationType,domainType,output_dir_path]
        simulation_parameters = [t_end,dt]
        state_variables = [variable_names,var_init_conds]
        #boundary_conditions = []
        mesh_settings = [meshDims]

        # form the mesh covering the computational domain
        mesh = UnitSquareMesh(nx, ny)

        # boundary conditions
        boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        boundary_markers.set_all(9999)

        bx0 = BoundaryX0()
        bx0.mark(boundary_markers, 0)
        bx1 = BoundaryX1()
        bx1.mark(boundary_markers, 1)
        by0 = BoundaryY0()
        by0.mark(boundary_markers, 2)
        by1 = BoundaryY1()
        by1.mark(boundary_markers, 3)

        # Redefine boundary integration measure
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
  
        # Collect Dirichlet conditions
        BCs = []
        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                bc = DirichletBC(V, [Constant(j) for j in boundary_conditions[i]['Dirichlet']],
                                boundary_markers, i)
                bcs.append(bc)
                
        # set up the test nad trial function spaces
        U,V = setup_function_space(mesh,variable_names,simulation_settings,state_variables)

        # set initial conditions
        #U,u_n = setup_initial_conditions(U,V,state_variables,simulation_settings)

        u_n = Function(V)
        u_n.interpolate(InitialConditions(IC = state_variables[1], Randomise = 1, DomainType = domainType))
        U.assign(u_n)

        # define measure across domain
        dx = Measure('dx', domain=mesh)

        measures = [dx,ds]

        F = 0

        # define test functions
        vTuple = TestFunctions(V)


        # Collect Neumann integrals
        for i in boundary_conditions:
            if 'Neumann' in boundary_conditions[i]:
                index=0
                for value in boundary_conditions[i]['Neumann']:
                    index +=1
                    if value != 0:
                        F += value*vTuple[index]*ds(i)
                

        # Simpler Robin integrals
        for i in boundary_conditions:
            if 'Robin' in boundary_conditions[i]:
                index=0
                for value in boundary_conditions[i]['Robin']:
                    index +=1
                    [r, s] = value
                    F += r*(u - s)*vTuple[index]*ds(i)

        print("HERe")
        # add the weak formulations of the separate PDEs
        F += general_weak_formulation(U,u_n,V,mesh,state_variables,simulation_settings,simulation_parameters,measures,diffusionValues,reactions,0)


        # add Newmann and Robin BC's, define dirichlet BC
        # boundary conditions
        #[F, boundary_conditions] = setup_boundary_conditions(F,U,U_N,V,boundary_markers,measures,state_variables,simulation_settings,boundary_conditions_vec)


        # Assemble the (non-)linear system 
        FA = lhs(F)
        Fb = rhs(F)
        A = assemble(FA, keep_diagonal = True)

        # Create VTK files for visualization output
        vtk_files = []
        for var_name in variable_names:
            vtk_files.append(File(str(output_dir_path+'/'+var_name+'_.pvd')))

            

        # run through the simulation step by step
        # Solving the problem in time
        t = t_initial

        plotSampleRate = 10
        n=0
        plotIndex=0

        while t< t_end:
            # Update time-step
            t += dt

            # Solve linear variational problem for time step
            b = assemble(Fb)
            solve(F==0,  U, BCs)  

            # Save solution to file (VTK)
            for var_name in variable_names:
                # specific variable solution specfied by the index
                index = variable_names.index(var_name)
                vtk_files[index] << (U.sub(index), t)

            # update the current variable vector for the next simulation
            u_n.assign(U)
            if(n==plotIndex*plotSampleRate):
                plot(U)
                plt.show()
                plotIndex=plotIndex+1
            n=n+1

        plt.show()
    except:
        print("Reaction_pde_solver: Fenics solver error")



    return output_dir_path


def General_ode_solver(source_term,variable_names,parameterList,times,var_init_conds):

    try:
  
        parameterNames,parameterValues = convert_list_to_tuples(parameterList)

        namesSymb = sym.symbols(variable_names, negative=False)
        paramsSymb = sym.symbols(parameterNames, negative=False)

        # use sympy to plot 0-dim results

        ydotSYM = sym.sympify(source_term)
        equilibria = sym.solve( (ydotSYM), namesSymb )
        print("Equlibria points: ",equilibria)


        t_final = times[1]
        t_initial=times[0]
        tout = np.linspace(t_initial, t_final)
        print("dt= ",(t_final-t_initial)/len(tout))
        t = sym.symbols('t') 
        f = sym.lambdify((namesSymb, t) + tuple(paramsSymb), ydotSYM)

        yout, info = odeint(f, var_init_conds, tout, parameterValues, full_output=True)
        plt.plot(tout, yout)
        plt.legend(variable_names)


    except:
        print("General_ode_solver(): solver error")

    return


def Reaction_ode_solver(reactions,times,var_init_conds):

    try:
  
        names, fenicsNames = determine_names_from_reactions(reactions)
        ydot, y, k = form_reaction_term(reactions, names)
        k_val_names,k_vals = retrieve_rate_constants(reactions)


        t_final = times[1]
        t_initial= times[0]
        tout = np.linspace(t_initial, t_final)
        print("dt= ",(t_final-t_initial)/len(tout))
        t = sym.symbols('t')  
        f = sym.lambdify((y, t) + k, ydot) 
        plt.plot(tout, odeint(f, var_init_conds, tout, tuple(k_vals)))
        plt.legend(names)

    except:
        print("Reaction_ode_solver(): solver error")

    return


def plotCellFunctions(cellSources,T,dt):
    
    tvec = np.linspace(0,T, int(T/dt))
    
    for cell in cellSources:
        
        values= [[0 for i in range(0,len(cell["function"]))] for j in range(0,len(tvec))]
        
        for i in range(0,len(values)):
            values[i] = ApplyCellSourceToValues(cell,values[i],tvec[i])
        
        uvec = [0 for i in range(0,len(values))]
        for i in range(0,len(values[0])):
            for j in range(0,len(values)):
                uvec[j]=values[j][i]
        
            plt.plot(tvec,uvec, label = "State "+str(i)+" "+cell["function"][i])
        
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.title("Source of Cell at "+"("+str(cell["location"][0])+","+str(cell["location"][1])+")")
        plt.show()

def plotBoxFunctions(BoxSources,T,dt):
    
    tvec = np.linspace(0,T, int(T/dt))
    
    for box in BoxSources:
        
        values= [[0 for i in range(0,len(box["function"]))] for j in range(0,len(tvec))]
        
        for i in range(0,len(values)):
            values[i] = ApplyCellSourceToValues(box,values[i],tvec[i])
        
        uvec = [0 for i in range(0,len(values))]
        for i in range(0,len(values[0])):
            for j in range(0,len(values)):
                uvec[j]=values[j][i]
        
            plt.plot(tvec,uvec, label = "State "+str(i)+" "+box["function"][i])
        
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("amplitude")
        if(len(box["location"])>2):
            plt.title("Source of box at "+"("+str(box["location"][0])+","+str(box["location"][2])+")")
        else:
            plt.title("Source of box at "+"("+str(box["location"][0])+","+str(box["location"][1])+")")
        plt.show()


class CellSourceAtX0(UserExpression):
    def __init__(self, eps, x0, t, degree, **kwargs):
        self.eps = eps
        self.x0 = x0
        self.t = t
        self.degree = degree
        UserExpression.__init__(self, **kwargs) 
    def eval(self, values, x):
        eps = self.eps
        t = self.t
        x0 = self.x0
        if((x[0]-x0[0])**2+(x[1]-x0[1])**2<eps):
            #print("before:" + str(values[0]))
            values[0] = OscilatorySource(t)#pulseSource(t,[0.5,1.5])
            #print("x:"+str(x[0])+" after:" + str(values[0]))
        else:
            values[0] = 0


    def value_shape(self): 
        return ()


def CheckIfAtCell(cellSources,x):
    cellIndex=0
    isAtCell = False
    for cell in cellSources:
        if((x[0]-cell["location"][0])**2+(x[1]-cell["location"][1])**2<cell["radius"]):
            isAtCell=True
            break
        cellIndex = cellIndex +1
        
    return isAtCell,cellIndex


def ApplyCellSourceToValues(cell,values,t):
    
    numberOfFunctions = len(cell["function"])
    
    # run through all the function defined for each of the state variables (values) i
    for i in range(0,numberOfFunctions):
        if(cell["function"][i]=="oscillatory"):
            values[i] = OscilatorySource(t,cell["parameters"][i])
        elif(cell["function"][i]=="pulse"):
            values[i] = PulseSource(t,cell["parameters"][i])
        elif(cell["function"][i]=="lagPulse"):
            values[i] = LagPulseSource(t,cell["parameters"][i])
        else:
            values[i] = 0
    
    # set all other state variables to 0
    for i in range(numberOfFunctions,len(values)):
        values[i] = 0
    
    return values


class CellSourceSet(UserExpression):
    def __init__(self, cellSources, t, degree, **kwargs):
        self.cellSources = cellSources
        self.t = t
        self.degree = degree
        UserExpression.__init__(self, **kwargs) 
    def eval(self, values, x):
        t = self.t
        cellSources = self.cellSources
        # values is the state variable vector at point x
        
        if(len(cellSources)>0):
            
            isAtCell, cellIndex = CheckIfAtCell(cellSources,x)
        
            if(isAtCell):
                values = ApplyCellSourceToValues(cellSources[cellIndex],values,t)
            else:
                for i in range(0,len(values)):
                    values[i] = 0
        else:
            for i in range(0,len(values)):
                values[i] = 0


    def value_shape(self): 
        return ()

class BoxSourceSet(UserExpression):
    def __init__(self, boxSources, t, degree, **kwargs):
        self.boxSources = boxSources
        self.t = t
        self.degree = degree
        UserExpression.__init__(self, **kwargs) 
    def eval(self, values, x):
        t = self.t
        boxSources = self.boxSources
        # values is the state variable vector at point x
        
        if(len(boxSources)>0):
            
            isInBox, boxIndex = CheckIfInBox(boxSources,x)
        
            if(isInBox):
                values = ApplyCellSourceToValues(boxSources[boxIndex],values,t)
            else:
                for i in range(0,len(values)):
                    values[i] = 0
        else:
            for i in range(0,len(values)):
                values[i] = 0


    def value_shape(self): 
        return ()

class GeometricSourceSet(UserExpression):
    def __init__(self, cellSources, boxSources, t, degree, **kwargs):
        self.cellSources = cellSources
        self.boxSources = boxSources
        self.t = t
        self.degree = degree
        self.numberOfSources = len(cellSources) + len(boxSources)
        UserExpression.__init__(self, **kwargs) 
    def eval(self, values, x):
        t = self.t
        cellSources = self.cellSources
        boxSources = self.boxSources
        # values is the state variable vector at point x
        isPointFound=False
        
        if(self.numberOfSources>0):
            
            if(len(cellSources)>0):
            
                isAtCell, cellIndex = CheckIfAtCell(cellSources,x)

                if(isAtCell):
                    values = ApplyCellSourceToValues(cellSources[cellIndex],values,t)
                    isPointFound=True
                
            if(isPointFound==False and len(boxSources)>0):
            
                isInBox, boxIndex = CheckIfInBox(boxSources,x)

                if(isInBox):
                    values = ApplyCellSourceToValues(boxSources[boxIndex],values,t)
                    isPointFound=True
                
            if(isPointFound==False):
                for i in range(0,len(values)):
                    values[i] = 0
        else:
            for i in range(0,len(values)):
                values[i] = 0


    def value_shape(self): 
        return ()

class K(UserExpression):
    def __init__(self, yValue, diffusionFactor, t, **kwargs):
        super().__init__(**kwargs)
        self.yValue = yValue
        self.diffusionFactor = diffusionFactor
        
    def eval(self, value, x):
        if x[1] <= self.yValue:
            value[0] = self.diffusionFactor
        else:
            value[0] = 1.0
    
    def value_shape(self): 
        return ()

def CheckIfInBox(boxSources,x):
    
    tol=1e-4
    
    boxIndex=0
    isInBox = False
    for box in boxSources:
        if(len(box["location"])>2):
            
            # the vertices need checking to ensure the correct ordering
            # take relative to the first vertex
            
            # cba, assume right now that it's the first (0) and third (2) vertex
            if(box["location"][0][0]<=x[0]<=box["location"][2][0] and box["location"][0][1]<=x[1]<=box["location"][2][1]):
                isInBox=True
                break
        else:
            # only the opposite corners are detailed, cba checking minor/major assume first (0) then second (1)
            if(box["location"][0][0]<=x[0]<=box["location"][1][0] and box["location"][0][1]<=x[1]<=box["location"][1][1]):
                isInBox=True
                break
                
        boxIndex = boxIndex +1
        
    return isInBox,boxIndex


def pHProbeSimulation(cellSources,boxSources,T,num_steps,dt,solutionFileName, mesh):
    solutionFileName = 'pHProbeSimulation/solution.pvd'
    # Create mesh and define function space

    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0), boundary)

    # Define initial value
    #u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
    #                 degree=2, a=0.001)
    u_0 = Constant(0)

    u_n = interpolate(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    f = GeometricSourceSet(t=0, cellSources=cellSources, boxSources=boxSources,degree=1)
    F = u*v*dx + 0.1*dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)


    # Create VTK file for saving solution
    vtkfile = File(solutionFileName)


    #plotting stuff
    numberRows = 4
    numberColumns = 4
    #figure, axes = plt.subplots(nrows=4, ncols=4)
    numberOfPlots = numberRows*numberColumns
    plotSampleRate = ceil(num_steps/numberOfPlots)

    # Time-stepping
    u = Function(V)
    t = 0
    plotIndex = 0
    plotRowIndex=0
    plotColumnIndex=0
    for n in range(num_steps):

        # Update current time
        t += dt
        # update user expression with the time
        f.t = t


        # Compute solution
        solve(a == L, u, bc)

        # Save to file and plot solution
        vtkfile << (u, t)

        if(n==plotIndex*plotSampleRate):


            plotIndex = plotIndex +1

            plot(u)
            plt.title("t="+str(t))
            plt.show()

        # Update previous solution
        u_n.assign(u)

    # Hold plot
    plt.show()              




def pHProbeCellSimulation(cellSources,boxSources,domain,subRegionByLocation,T,num_steps,dt,solutionFileName, mesh):

    solutionFileName = 'pHProbeSimulation/solution.pvd'
    # Create mesh and define function space

    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary

    class K(UserExpression):
        def __init__(self, yValue, diffusionFactor, t, **kwargs):
            super().__init__(**kwargs)
            self.yValue = yValue
            self.diffusionFactor = diffusionFactor
            
        def eval(self, value, x):
            if x[1] <= self.yValue:
                value[0] = self.diffusionFactor
            else:
                value[0] = 1.0
        
        def value_shape(self): 
            return ()


    bc = DirichletBC(V, Constant(0), boundary)

    # Define initial value
    #u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
    #                 degree=2, a=0.001)
    u_0 = Constant(0)

    u_n = interpolate(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    K = K(yValue=subRegionByLocation[0]["yValue"],diffusionFactor=subRegionByLocation[0]["diffusionFactor"], t=0)

    f = GeometricSourceSet(t=0, cellSources=cellSources, boxSources=boxSources,degree=1)
    F = u*v*dx + domain[0]["diffusionCoeffs"]*K*dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)


    # Create VTK file for saving solution
    vtkfile = File(solutionFileName)


    #plotting stuff
    numberRows = 4
    numberColumns = 4
    #figure, axes = plt.subplots(nrows=4, ncols=4)
    numberOfPlots = numberRows*numberColumns
    plotSampleRate = ceil(num_steps/numberOfPlots)

    # Time-stepping
    u = Function(V)
    t = 0
    plotIndex = 0
    plotRowIndex=0
    plotColumnIndex=0
    for n in range(num_steps):

        # Update current time
        t += dt
        # update user expression with the time
        f.t = t


        # Compute solution
        solve(a == L, u, bc)

        # Save to file and plot solution
        vtkfile << (u, t)

        if(n==plotIndex*plotSampleRate):


            plotIndex = plotIndex +1

            plot(u)
            plt.title("t="+str(t))
            plt.show()

        # Update previous solution
        u_n.assign(u)

    # Hold plot
    plt.show()            