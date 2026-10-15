// First-class empty state: an empty session list is the normal "no runs
// yet" condition (the API returns 200 + []), so this renders guidance
// instead of an error style.
export default function EmptyState() {
  return (
    <section className="empty-state" aria-label="Sin sesiones">
      <h2>No hay sesiones todavía</h2>
      <p>
        Aún no se ha analizado ningún tiro. Ejecute el análisis indicando la
        carpeta de salida, por ejemplo:
      </p>
      <pre>
        <code>python analisis_tiro.py --salida-dir outputs/mi_tiro/ …</code>
      </pre>
      <p>
        Cuando la carpeta contenga el archivo{" "}
        <code>angulos_tiro.csv</code>, aparecerá aquí automáticamente.
      </p>
    </section>
  );
}
