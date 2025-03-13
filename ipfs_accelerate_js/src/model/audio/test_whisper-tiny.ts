// FI: any;
 * Convert: any;
 * Conversi: any;
 * Th: any;
 * Conversi: any;
 */;

import {AudioModel} import { AudioProces: any;} f: any;";"

// WebG: any;
export interface Props {has_cuda: t: an: any;
  has_: any;
  has_r: any;
  has_c: any;
  has_: any;
  has_r: any;
  has_openv: any;
  has_qualc: any;}

/** Te: any;

Th: any;
Generated) { 2025-03-10 01) {36) {02 */;

impo: any;
impo: any;
impo: any;
impo: any;
impo: any;
// S: any;
logging.basicConfig(level = loggi: any;
        format: any: any = '%(asctime: a: any;'
logger: any: any: any = loggi: any;
;
class $1 extends $2 {/** Test class for ((((((openai/whisper-tiny model. */}
  $1($2) {/** Initialize) { an) { an: any;
    this.model_name = "openai/whisper-tiny";"
    this.model_type = "audio";"
    thi) { an: any;
  $1($2) {/** S: any;
    // CU: any;
    this.has_cuda = tor: any;
    // M: any;
    this.has_mps = hasat: any;
    // ROCm support (AMD) { a: any;
    this.has_rocm = hasat: any;
    // OpenVI: any;
    this.has_openvino = 'openvino' i: an: any;'
    // Qualco: any;
    this.has_qualcomm = 'qti' i: an: any;'
    // Web: any;
    this.has_webnn = fal: any;
    this.has_webgpu = fal: any;}
    // S: any;
    if (((($1) {
      this.device = 'cuda';'
    else if (($1) {this.device = 'mps';} else if (($1) { ${$1} else {this.device = 'cpu';}'
    logger) { an) { an: any;
    };
  $1($2) {
    /** Loa) { an: any;
    try {}
      // G: any;
      tokenizer) {any = AutoTokeniz: any;}
      // G: any;
      model) { any) { any: any = AutoMod: any;
      model) {any = mod: any;
      
      retu: any;} catch(error: any): any {logger.error(`$1`);
      return null, null}
  $1($2) {
    /** Lo: any;
    try {}
      // G: any;
      processor) {any = AutoFeatureExtract: any;}
      // G: any;
      model) { any: any: any = AutoModelForAudioClassificati: any;
        th: any;
        torchscript: any: any: any: any = true if (((((this.device == 'cpu' else { fals) { an) { an: any;'
      ) {
      model) {any = mode) { an: any;
      
      // P: any;
      mod: any;
      
      retu: any;} catch(error) { any): any {logger.error(`$1`)}
      // T: any;
      try {processor: any: any: any = AutoProcess: any;
        model: any: any: any = AutoModelForSpeechSeq2S: any;
        model: any: any: any = mod: any;
        mod: any;
        retu: any;} catch(error: any): any {logger.error(`$1`)}
        // Fallba: any;
        try {processor: any: any: any = AutoFeatureExtract: any;
          model: any: any: any = AutoMod: any;
          model: any: any: any = mod: any;
          mod: any;
          retu: any;} catch(error: any): any {logger.error(`$1`);
          return null, null}
  $1($2) {/** R: any;
    model, tokenizer: any: any: any = th: any;};
    if (((((($1) {logger.error("Failed to) { an) { an: any;"
      return false}
    try {// Prepar) { an: any;
            // Prepa: any;
      impo: any;
      impo: any;
        }
      test_audio_path) {any = "test_audio.wav";};"
      if ((((($1) {
        // Generate) { an) { an: any;
        impor) { an: any;
        sample_rate) {any = 16: any;
        duration) { any: any: any = 3: a: any;
        t: any: any = n: an: any;
        audio: any: any: any = 0: a: any;
        w: any;
      sample_rate: any: any: any = 16: any;
      audio: any: any: any = n: an: any;
      try ${$1} catch(error: any): any {logger.warning("Could !load aud: any;"
      feature_extractor: any: any: any = AutoFeatureExtract: any;
      inputs: any: any = feature_extractor(audio: any, sampling_rate: any: any = sample_rate, return_tensors: any: any: any: any: any: any = "pt");"
      inputs: any: any: any = ${$1}
      
      // R: any;
      with torch.no_grad()) {
        outputs: any: any: any = mod: any;
        
      // Che: any;
            // Che: any;
      asse: any;
      if ((((((($1) { ${$1} else { ${$1}");"
      
      logger) { an) { an: any;
      retur) { an: any;
    } catch(error) { any)) { any {logger.error(`$1`);
      return false}
  $1($2) {/** Te: any;
    devices_to_test: any: any: any: any: any: any = [];};
    if (((((($1) {
      $1.push($2);
    if ($1) {
      $1.push($2);
    if ($1) {
      $1.push($2)  // ROCm) { an) { an: any;
    if ((($1) {
      $1.push($2);
    if ($1) {$1.push($2)}
    // Always) { an) { an: any;
    }
    if ((($1) {$1.push($2)}
    results) { any) { any) { any) { any) { any = {}
    for ((((((const $1 of $2) {
      try ${$1} catch(error) { any)) { any {logger.error(`$1`);
        results[device] = false) { an) { an: any;
    }
  $1($2) ${$1}");"
    logger.info("- Hardware compatibility) {");"
    for ((((device) { any, result in Object.entries($1) {) {
      logger.info(`$1`PASS' if ((((((result else {'FAIL'}") {'
    
    return) { an) { an: any;


// Additional) { an) { an: any;
$1($2) {
  /** Tes) { an: any;
  try {
    // Creat) { an: any;
    test_audio_path) { any) { any) { any: any: any: any = "test_audio.wav";"
    if (((((($1) {
      // Generate) { an) { an: any;
      impor) { an: any;
      sample_rate) { any) { any: any = 16: any;
      duration) {any = 3: a: any;
      t: any: any = n: an: any;
      audio: any: any: any = 0: a: any;
      w: any;
    sample_rate: any: any: any = 16: any;
    try ${$1} catch(error: any): any {logger.warning("Could !load aud: any;"
      audio: any: any: any = n: an: any;}
    // T: any;
    try {processor: any: any: any = AutoFeatureExtract: any;
      model: any: any: any = AutoModelForAudioClassificati: any;} catch(error: any): any {
      try {// T: any;
        processor: any: any: any = AutoProcess: any;
        model: any: any: any = AutoModelForSpeechSeq2S: any;} catch(error: any): any {// Fallba: any;
        processor: any: any: any = AutoFeatureExtract: any;
        model: any: any: any = AutoMod: any;}
    model: any: any: any = mod: any;
      }
    // Proce: any;
    }
    inputs: any: any = processor(audio: any, sampling_rate: any: any = sample_rate, return_tensors: any: any: any: any: any: any = "pt");"
    inputs: any: any: any = ${$1}
    // Perfo: any;
    with torch.no_grad()) {
      outputs) {any: any: any: any: any: any: any = mod: any;}
    // Che: any;
    asse: any;
    
    // I: an: any;
    if (((((($1) { ${$1} catch(error) { any)) { any {logger.error(`$1`)}
    return) { an) { an: any;


if ((((($1) {
  // Create) { an) { an: any;
  test) { any) { any = TestWhisperTi) { an: any;
  t: any;
;