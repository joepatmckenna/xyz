import { darkMode } from '$lib/stores.js';

export const load = async ({ data }) => {
	darkMode.set(data.darkMode);
};

export const prerender = true;
