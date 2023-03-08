import matplotlib.pyplot as plt


def grafica1(df):
    # Crear una lista de colores para los puntos
    colors = ['blue' if x == -1 else 'red' for x in df['status']]

    # Crear el gráfico de dispersión
    plt.scatter(df['ratio_intMedia'], df['ratio_intHyperlinks'], c=colors)

    # Agregar etiquetas de los ejes x e y
    plt.xlabel('ratio_intMedia')
    plt.ylabel('ratio_intHyperlinks')

    # Mostrar el gráfico
    plt.show(block=True)

def grafica2(df):
    # Crear una lista de colores para los puntos
    colors = ['blue' if x == -1 else 'red' for x in df['status']]

    # Crear el gráfico de dispersión
    plt.scatter(df['page_rank'], df['ratio_extMedia'], c=colors)

    # Agregar etiquetas de los ejes x e y
    plt.xlabel('page_rank')
    plt.ylabel('ratio_extMedia')

    # Mostrar el gráfico
    plt.show(block=True)

def grafica3(df):
    # Crear una lista de colores para los puntos
    colors = ['blue' if x == -1 else 'red' for x in df['status']]

    # Crear el gráfico de dispersión
    plt.scatter(df['ratio_intHyperlinks'], df['page_rank'], c=colors)
    
    # Agregar etiquetas de los ejes x e y
    plt.xlabel('ratio_intHyperlinks')
    plt.ylabel('page_rank')

    # Mostrar el gráfico
    plt.show(block=True)

    
    
def grafica4(df):
    # Crear una lista de colores para los puntos
    colors = ['blue' if x == -1 else 'red' for x in df['status']]

    # Crear el gráfico de dispersión
    plt.scatter(df['ratio_intMedia'], df['ratio_extMedia'], c=colors)
    
    # Agregar etiquetas de los ejes x e y
    plt.xlabel('ratio_intMedia')
    plt.ylabel('ratio_extMedia')

    # Mostrar el gráfico
    plt.show(block=True)
