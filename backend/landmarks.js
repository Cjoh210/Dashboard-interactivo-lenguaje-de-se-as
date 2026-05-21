// load-reference-landmarks.js
async function loadReferenceLandmarks() {
    try {
        const response = await fetch('/reference-landmarks');
        if (!response.ok) {
            throw new Error('Error al cargar los landmarks de referencia');
        }
        return await response.json();
    } catch (error) {
        console.error(error);
    }
}

// Cargar los landmarks de referencia y almacenarlos en una variable
let referenceLandmarks = [];

loadReferenceLandmarks().then(data => {
    referenceLandmarks = data;
});