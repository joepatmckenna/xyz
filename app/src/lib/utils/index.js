import { writable } from 'svelte/store';

export const darkMode = writable(false);

export const fetchPosts = async () => {
	const files = import.meta.glob('/src/routes/posts/*.md');
	return await Promise.all(
		Object.entries(files).map(async ([path, load]) => {
			const content = await load();
			return {
				path: path.slice('/src/routes'.length, -'.md'.length),
				metadata: content.metadata,
				html: content.default.$$render()
			};
		})
	);
};
