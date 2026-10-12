const form = document.querySelector('#settings');
let appliedSettings = JSON.parse(document.querySelector('#applied-settings').textContent);
const status = document.querySelector('#form-status');
const download = document.querySelector('#download');
const submit = form.querySelector('[type="submit"]');
function settingsFromForm() {
  return {
    estimator: form.elements.estimator.value,
    portfolio_value: Number(form.elements.portfolio_value.value),
    concentration_limit: Number(form.elements.concentration_limit.value) / 100,
    scenario_loss_limit: Number(form.elements.scenario_loss_limit.value) / 100,
    stale_calendar_days: Number(form.elements.stale_calendar_days.value),
    scenario: form.elements.scenario.value,
    weights: [...form.querySelectorAll('[name="weight"]')].map(input => Number(input.value) / 100),
    shocks: [...form.querySelectorAll('[name="shock"]')].map(input => Number(input.value) / 100)
  };
}
function message(text, error = false) {
  status.textContent = text;
  status.classList.toggle('error', error);
}
form.addEventListener('input', () => message('Settings changed. Apply to update results; export still uses applied settings.'));
form.addEventListener('submit', async event => {
  event.preventDefault();
  submit.disabled = download.disabled = true;
  message('Recalculating…');
  try {
    const response = await fetch('/api/analyze', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(settingsFromForm())});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Calculation failed');
    document.querySelector('#analysis').innerHTML = data.html;
    appliedSettings = data.settings;
    message(`Applied. ${data.exceptions} items require review.`);
  } catch (error) {
    message(`${error.message}. Results still show the previous applied settings.`, true);
  } finally { submit.disabled = download.disabled = false; }
});
document.querySelector('#preset').addEventListener('change', event => {
  const preset = event.target.value;
  const shocks = preset === 'market' ? [-10, -10, -10, -10, -10] : preset === 'tech' ? [-20, -15, -10, -5, -5] : null;
  if (shocks) form.querySelectorAll('[name="shock"]').forEach((input, index) => input.value = shocks[index]);
  document.querySelector('#scenario-rationale').textContent = preset === 'market' ? 'Tests a uniform equity repricing. A fully invested portfolio loses 10% regardless of basket allocation.' : preset === 'tech' ? 'Tests whether technology-linked baskets dominate losses when the rest of the market falls less.' : 'Enter a total simple return for each basket and document the economic interpretation when presenting it.';
  message('Scenario changed. Apply to update results.');
});
form.querySelectorAll('[name="shock"]').forEach(input => input.addEventListener('input', () => {
  document.querySelector('#preset').value = 'custom';
  document.querySelector('#scenario-rationale').textContent = 'Custom total basket returns; inspect each shock before interpreting the resulting P&L.';
}));
form.elements.scenario.addEventListener('change', () => {
  document.querySelector('#shock-controls').hidden = form.elements.scenario.value !== 'hypothetical';
  message('Scenario changed. Apply to update results.');
});
document.querySelector('#reset').addEventListener('click', () => location.reload());
download.addEventListener('click', async () => {
  download.disabled = true;
  try {
    const response = await fetch('/api/report', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(appliedSettings)});
    if (!response.ok) throw new Error((await response.json()).error || 'Export failed');
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = (response.headers.get('Content-Disposition') || '').match(/filename="([^"]+)"/)?.[1] || 'quiet-index-report.html';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message('Exported the applied analysis as a self-contained HTML report.');
  } catch (error) { message(error.message, true); }
  finally { download.disabled = false; }
});
