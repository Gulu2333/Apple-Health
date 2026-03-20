<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Apple Health Sleep &amp; Physiology Analyzer – Description and Usage</title>
</head>
<body>
  <h1>Apple Health Sleep &amp; Physiology Analyzer – Description and Usage</h1>

  <h2>Overview</h2>
  <p>
    This script analyzes an Apple Health <code>export.xml</code> file and produces a complete set of
    publication‑ready figures (APA 7th styling) and a structured text report about your sleep and related
    physiological signals. It focuses on sleep stages, heart rate, heart rate variability (HRV), and blood
    oxygen saturation (SpO2) during sleep.
  </p>

  <h2>Main Features</h2>
  <ul>
    <li>Reads and stream‑parses the large Apple Health <code>export.xml</code> file efficiently using an iterative XML parser.</li>
    <li>Extracts sleep stage records, heart rate, HRV (SDNN), and SpO2 records from Apple Health data.</li>
    <li>Merges fragmented sleep intervals into full “nights,” filters out short naps, and summarizes valid sleep sessions.</li>
    <li>Aligns heart rate, HRV, and SpO2 data with sleep stages and with time since sleep onset to study changes across the night.</li>
    <li>Identifies and analyzes sleep stage transitions and their associated physiological changes.</li>
    <li>Bins and smooths physiological data in fixed time windows to create regression curves over the course of sleep.</li>
    <li>Automatically generates figures for data quality, sleep regression, circadian rhythm, sleep stages, transitions, hypnograms, and stage proportions.</li>
    <li>Applies APA 7th‑style plotting defaults for all exported figures.</li>
    <li>Writes a descriptive analysis report (<code>Analysis_Report.txt</code>) that summarizes key results and references the generated figures.</li>
  </ul>

  <h2>Input Data Requirements</h2>
  <ul>
    <li>An Apple Health export created on an iPhone or Apple Watch device, saved as <code>export.xml</code>.</li>
    <li>The file must be a valid XML export from Apple Health; the script checks that the provided path points to an <code>.xml</code> file.</li>
  </ul>

  <h2>Software Requirements</h2>
  <p>Install these Python packages before running the script:</p>
  <ul>
    <li><code>pandas</code></li>
    <li><code>matplotlib</code></li>
    <li><code>seaborn</code></li>
    <li><code>scipy</code></li>
    <li><code>numpy</code></li>
  </ul>

  <h2>How to Run</h2>
  <ol>
    <li>Open a terminal or command prompt.</li>
    <li>Change directory to the folder containing <code>apple_health_analyzer_2.py</code>.</li>
    <li>Run <code>python apple_health_analyzer_2.py</code>.</li>
    <li>When prompted, provide the path to <code>export.xml</code> (drag‑and‑drop or paste the path) and press Enter.</li>
    <li>Follow the on‑screen prompts for optional date‑range and hypnogram settings, and wait for the analysis to finish.</li>
  </ol>

  <h2>Output</h2>
  <ul>
    <li>An <code>output</code> directory created next to the script.</li>
    <li><code>Analysis_Report.txt</code> – the main textual summary of your Apple Health sleep and physiology analysis.</li>
    <li>Multiple PNG figures (for example, <code>Figure_01_*.png</code>, <code>Figure_02_*.png</code>) showing data quality, sleep regression, circadian rhythm, stages, transitions, hypnograms, and stage proportions.</li>
  </ul>

  <h2>Notes</h2>
  <ul>
    <li>The script is intended for personal research and academic‑style reporting, not medical diagnosis or treatment.</li>
    <li>Large <code>export.xml</code> files can take several minutes to parse and analyze.</li>
    <li>For best results, export a sufficiently long date range from Apple Health so that there are multiple nights and enough physiological data.</li>
  </ul>
</body>
</html>
