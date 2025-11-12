/**
 * Retrieve ENV variable or throw Error
 * @param name The name of the ENV variable
 * @param defaultValue The default value (optional)
 */
export const getOrThrow = (name: string, defaultValue?: string) => {
  if (defaultValue) {
    return defaultValue;
  }
  const value = process.env[name];
  if (!value) throw new Error(`${name} Environment variable is missing`);
  return value;
};
