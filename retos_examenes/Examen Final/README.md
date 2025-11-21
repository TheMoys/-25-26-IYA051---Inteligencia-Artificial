# Trabajo para el Examen Final de Asignatura: 051 - Inteligencia Artificial

## Reto del examen final
El trabajo consistirá en la realización de un software OCR con las siguientes características:

- **Input:** archivo externo en formato Imagen: jpg, bmp, tiff, png… (mínimo 1 formato)  
  *(sin texto embebido asociado)*

- **Output:** 1 o varios archivos según lo que resuelva el software de entre las siguientes funcionalidades:

  - **Funcionalidades obligatorias** (se valorará en función del rendimiento):
    - Poder escanear imagen con caracteres digitales (tipografía de imprenta) y convertir a texto digital.
    - Poder escanear imagen con caracteres manuales (tipografía de escritura manual) y convertir a texto digital.

  - **Funcionalidades opcionales** (se valorará como extra a las principales):
    - Identificar imágenes y guardar las imágenes encuadradas como archivo independiente.
    - Identificar tablas y digitalizarlas a formato tabla (se puede dar el formato con algún lenguaje como HTML, LaTeX, Markdown, etc., dibujando las líneas o simplemente creando una estructura nodos y conexiones).
    - Detectar marcas, tags, códigos de barra, códigos QR… (no hace falta interpretarlos), sí guardarlos en archivos de imagen independiente.
    - Detectar alguna característica en imágenes para clasificarlas (ojos, caras, manos, elementos de paisajes, etc.).
    - Características extra propuestas por el alumno: **preguntar al profesor**.

---

## Normas de ejecución

Se deberá resolver a nivel individual.

Se podrán abordar de forma colaborativa ciertas tareas, en la que todos los alumnos están obligados a participar (siempre que las peticiones no sean desmesuradas).

La obligatoriedad de participación tiene fecha límite: **15 de Diciembre**. Por ejemplo: necesito entrenar una red con distintas grafías manuscritas, como alumno para mi trabajo individual puedo solicitar que cada uno de mis compañeros escriba en una hoja 5 veces el abecedario y 100 palabras de mi elección según distintas pautas que yo considere (no llevaría más de 10 min).

Se pueden usar todas las tipografías, datasets de imágenes que se estime necesario.

**Restricción:** No es posible el uso de librerías o aplicaciones externas que ya resuelvan el problema del reconocimiento de imágenes, detección de texto, palabras, letras o características, etc. similares a *“Tesseract OCR”*, PERO sí podéis informaros cómo afrontan los distintos problemas para crear vuestra propia solución.

Los alumnos podrán definir requisitos de formato input para el funcionamiento de su código, por ejemplo:  
imágenes de no más de…,  
imágenes en formato …,  
embebidas en un pdf,  
un marco rectangular en color negro puro con 1mm de grosor alrededor de lo que se quiere escanear,  
tamaño de letra máximo o mínimo,  
rangos de orientación del texto, etc.

---

# Estructura del trabajo a presentar

## Trabajo escrito

### 1.- Introducción
El trabajo escrito deberá presentar una primera parte explicando cómo se ha afrontado el problema (de 3 a 9 páginas de extensión, si se necesitan más se puede todas las que se quiera).

### 2.- Solución aportada [mínimo 3 páginas]
Segunda parte con el código bien comentado (al más mínimo nivel de detalle de los inputs de cada función justificando su valor).  
Habrá un apartado de análisis en el que se mida y valore el rendimiento de la solución aportada.

En el caso de algoritmos indicar estructuras, flujogramas, etc.

### 3.- Registro de Resultados [extensión que se quiera]
Registro de las tomas de decisión y resultados del proceso evolutivo del proyecto.

Con fecha y hora:  
Prueba. Resultados. Anotaciones.

### 4.- Conclusiones [1–2 páginas]
Apartado final de posibles mejoras y conclusiones.

---

## Código

Puede ser en IDE (.py), o Jupyter/GoogleColab (.ipynb).

En caso de ser necesarias librerías externas para ejecución, indicar:  
versión, fuente, descarga e instalación requeridas.

---

## Vídeo

Cada alumno deberá realizar una presentación en formato vídeo presentando el trabajo realizado y demostrando que funciona.  
El alumno debe aparecer con la cara descubierta y de forma que sea fácilmente identificable (no es necesaria la aparición durante todo el vídeo: 30 seg. mínimo al inicio como presentación es suficiente).

- 1 min de presentación del trabajo.
- 3–5 min de la solución aportada y diferentes problemas superados o alternativas tomadas.
- 3–5 min de ejecución de la solución aportada con distintos ejemplos a selección del alumno.  
  *(Debe indicarse cómo es el procedimiento para adjuntar imágenes para que el profesor pueda probarlo con imágenes propias).*
