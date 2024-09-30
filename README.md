# IA-IOT

##Práctica 
**detección de malware en cámaras de seguridad conectadas a IoT mediante IA**
---

### **Paso 1: Explicación Simple del Proyecto**
**Objetivo**: Vamos a simular cómo una **cámara de seguridad IoT** puede ser atacada por un malware y cómo podemos usar la **Inteligencia Artificial (IA)** para detectarlo.

### Materiales:
- Un ordenador con internet.
- **Google Colab** (para escribir código).
- **Wireshark** (para capturar el tráfico de red).
- Simulación de tráfico de una **cámara IoT**.
  
### **Paso 2: Configurando Google Colab para la IA**
#### ¿Qué es Google Colab?
Google Colab es como un cuaderno en línea donde puedes escribir y ejecutar código sin necesidad de instalar nada en tu ordenador. Usaremos esto para programar nuestra IA.

#### **Instrucciones**:
1. Ve a [Google Colab](https://colab.research.google.com/).
2. Haz clic en "Nuevo cuaderno".
3. En la primera celda, copia y pega el siguiente código para configurar la IA que analizará el tráfico de la cámara en busca de patrones sospechosos.

```python
# Paso 1: Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paso 2: Generamos datos simulados para tráfico normal y tráfico con malware
# Creamos datos aleatorios que simulan el comportamiento de una cámara IoT
# "1" indica tráfico con malware, "0" indica tráfico normal

data = pd.DataFrame({
    'paquetes_por_segundo': np.random.randint(100, 1000, 1000),
    'tamaño_paquete': np.random.randint(200, 1500, 1000),
    'bandwidth': np.random.randint(50, 500, 1000),
    'malware': np.random.choice([0, 1], 1000)  # 0 = no hay malware, 1 = hay malware
})

# Paso 3: Dividimos los datos en entrenamiento y prueba
X = data[['paquetes_por_segundo', 'tamaño_paquete', 'bandwidth']]
y = data['malware']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 4: Entrenamos el modelo de IA para detectar malware
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Paso 5: Predecimos con los datos de prueba
y_pred = modelo.predict(X_test)

# Mostramos el resultado de la predicción
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo para detectar malware: {accuracy * 100:.2f}%")
```

#### **Explicación para los niños**:
- Este código está entrenando un **cerebro artificial** (modelo de IA) para aprender a distinguir entre el tráfico normal de la cámara y el tráfico con **malware** (software malo).
- Estamos utilizando datos simulados, pero en la realidad, este cerebro podría analizar el tráfico de red real.

---

### **Paso 3: Captura de tráfico con Wireshark**
#### ¿Qué es Wireshark?
Wireshark es una herramienta que te permite ver todo lo que está sucediendo en la red, como si fuera una lupa para ver cómo los dispositivos, como las cámaras de seguridad, envían y reciben información.

#### **Instrucciones**:
1. Descarga e instala **Wireshark** desde [wireshark.org](https://www.wireshark.org/).
2. Abre Wireshark y selecciona la red Wi-Fi a la que estés conectado.
3. Haz clic en "Start" para comenzar a capturar el tráfico de red.
4. Observa los paquetes que empiezan a llegar. ¡Estos son los mensajes que tu ordenador y otros dispositivos están enviando y recibiendo!

#### **Explicación para los niños**:
- Imagínate que todos los dispositivos de tu casa están hablando entre sí. Wireshark te permite **escuchar** esas conversaciones.
- Algunos de estos mensajes pueden ser buenos, pero otros podrían ser sospechosos, como si un extraño intentara entrar en la conversación.

---

### **Paso 4: Simulación de un Ataque en una Cámara IoT**
#### **Instrucciones**:
Para simular tráfico de malware en una cámara IoT, puedes usar herramientas como **Packet Sender** o simplemente simular un ataque básico en la red con un tráfico anómalo. Para esta práctica, explicamos el concepto de ataque, pero mantenemos la simulación simple para los niños.

1. **Explicación para los niños**:
   - Imaginen que hay un ladrón (malware) intentando espiar lo que está viendo la cámara de seguridad.
   - Vamos a enseñarle al **cerebro artificial** cómo detectar esos mensajes extraños y alertarnos cuando algo malo esté pasando.

2. **Ejercicio Simulado**:
   - Durante la captura en Wireshark, observamos qué pasa cuando hay mucho tráfico de red desde y hacia la cámara.
   - Compararemos este tráfico con lo que espera nuestra IA, que ya sabe cómo debería comportarse la cámara.

---

### **Paso 5: Probar la Detección de Malware en la IA**
#### **Instrucciones**:
Ahora volvamos a **Google Colab** y probemos nuestra IA con el tráfico capturado.

1. Copia el archivo de tráfico capturado de Wireshark en formato **CSV** y súbelo a Google Colab.
2. En la celda siguiente de Google Colab, copia este código para probar la detección en el tráfico capturado:

```python
# Paso 6: Cargamos el tráfico capturado desde Wireshark
# Subir el archivo CSV generado con Wireshark al entorno de Colab
# El archivo debe contener las columnas: 'paquetes_por_segundo', 'tamaño_paquete', 'bandwidth'

from google.colab import files
uploaded = files.upload()

# Paso 7: Leemos el archivo subido
import io
captured_data = pd.read_csv(io.BytesIO(uploaded['nombre_del_archivo.csv']))

# Paso 8: Predecimos si hay malware en el tráfico capturado
predicciones = modelo.predict(captured_data)

# Mostramos las predicciones
print("Predicciones del tráfico capturado (0 = No hay malware, 1 = Hay malware):")
print(predicciones)
```

#### **Explicación para los niños**:
- Subimos los datos capturados con Wireshark, y nuestro cerebro artificial (IA) nos dirá si hay **malware** en esos mensajes que vimos antes.

---

### **Paso 6: Reflexión y Resultados**
Después de ejecutar el código en Google Colab:
- Si el modelo detecta **1 (malware)**, podemos explicarle a los alumnos que se ha encontrado algo sospechoso en los mensajes de la cámara.
- Si todos los resultados son **0 (sin malware)**, significa que todo está en orden.

#### **Conclusión para los niños**:
¡Felicidades! Han utilizado **IA y una herramienta de red** para detectar posibles amenazas en una cámara IoT. Esta tecnología ayuda a protegernos de personas que intentan acceder a nuestros dispositivos sin permiso.

---

### **Paso 7: Tareas Opcionales**
- Investigar cómo mejorar la seguridad de una cámara IoT.
- Probar con otro dispositivo IoT (como un termostato o bombilla inteligente).


# **Código Completo en Google Colab de la Sesión**:

```python
# Paso 1: Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from google.colab import files
import io

# Paso 2: Generamos datos simulados para entrenar el modelo de IA
# Creamos datos simulados que representan el comportamiento de una cámara IoT
data = pd.DataFrame({
    'paquetes_por_segundo': np.random.randint(100, 1000, 1000),
    'tamaño_paquete': np.random.randint(200, 1500, 1000),
    'bandwidth': np.random.randint(50, 500, 1000),
    'malware': np.random.choice([0, 1], 1000)  # 0 = no hay malware, 1 = hay malware
})

# Dividimos los datos en entrenamiento y prueba
X = data[['paquetes_por_segundo', 'tamaño_paquete', 'bandwidth']]
y = data['malware']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamos el modelo de IA (RandomForestClassifier)
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Predecimos con los datos de prueba y evaluamos la precisión
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo para detectar malware: {accuracy * 100:.2f}%")

# Paso 3: Cargamos el tráfico capturado desde Wireshark
# Subimos el archivo CSV capturado con Wireshark al entorno de Colab

uploaded = files.upload()

# Leemos el archivo CSV subido
captured_data = pd.read_csv(io.BytesIO(uploaded['nombre_del_archivo.csv']))

# Mostramos las primeras filas para verificar los datos cargados
print("Datos capturados:")
print(captured_data.head())

# Paso 4: Filtramos el tráfico relevante (protocolos como HTTP, TCP, UDP)
protocolos_iot = ['HTTP', 'TCP', 'UDP']
iot_traffic = captured_data[captured_data['Protocol'].isin(protocolos_iot)]

# Paso 5: Añadimos columnas simuladas para 'paquetes_por_segundo' y 'bandwidth'
# Simulamos estos valores para que coincidan con las características que espera el modelo
iot_traffic['paquetes_por_segundo'] = iot_traffic['Length'] // 10  # Simulación
iot_traffic['bandwidth'] = iot_traffic['Length'] * 8  # Simulación del ancho de banda

# Ahora seleccionamos las 3 características que espera el modelo
X_nuevo = iot_traffic[['paquetes_por_segundo', 'Length', 'bandwidth']]

# Normalizamos los valores antes de hacer predicciones
scaler = StandardScaler()
X_nuevo_scaled = scaler.fit_transform(X_nuevo)

# Paso 6: Predecimos si hay malware en el tráfico capturado
predicciones = modelo.predict(X_nuevo_scaled)

# Mostramos las predicciones
print("Predicciones del tráfico capturado (0 = No hay malware, 1 = Hay malware):")
print(predicciones)

# Paso 7: Mostrar los resultados junto con las direcciones IP origen y destino
iot_traffic['Predicción Malware'] = predicciones
print(iot_traffic[['Source', 'Destination', 'Protocol', 'Length', 'Predicción Malware']])
```

### **Descripción del Código:**

1. **Entrenamiento del Modelo**:
   - Se crea un conjunto de datos simulado que incluye tres características: `paquetes_por_segundo`, `tamaño_paquete` y `bandwidth`.
   - Se entrena un modelo de **RandomForestClassifier** con este conjunto de datos para que aprenda a detectar si hay **malware** (0 = no hay malware, 1 = hay malware).
   - Se evalúa la precisión del modelo para asegurarnos de que funciona bien.

2. **Carga del Archivo Capturado con Wireshark**:
   - Se sube un archivo CSV capturado con **Wireshark** a Google Colab.
   - Se filtran los protocolos relevantes para dispositivos IoT (HTTP, TCP, UDP).

3. **Simulación de Características Faltantes**:
   - Dado que el archivo de **Wireshark** solo tiene columnas como `Length` (tamaño de los paquetes), se simulan las columnas `paquetes_por_segundo` y `bandwidth` para cumplir con las tres características que el modelo espera.

4. **Predicción de Malware**:
   - El modelo de IA predice si el tráfico contiene malware o no basándose en las características extraídas.
   - Se imprimen las predicciones y se muestran junto a las direcciones IP de origen y destino para identificar qué tráfico es sospechoso.

### **Instrucciones para Ejecutar**:
1. **Sube el archivo CSV capturado** con Wireshark a Google Colab.
2. Ejecuta el código paso por paso en Google Colab.
3. Observa los resultados y revisa si el modelo ha detectado posibles señales de malware en el tráfico de tu cámara IoT.

### **Notas Finales**:
- El código está diseñado para ser ejecutado en **Google Colab** y utiliza datos simulados para complementar la información faltante en el archivo capturado.
- Si tienes datos más específicos (como `paquetes_por_segundo` y `bandwidth`), puedes adaptarlo fácilmente eliminando la simulación de estas columnas.

